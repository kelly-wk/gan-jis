"""Deterministic smoke test and small/full recovered-data training entry points."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .core import (
    JISTables,
    best_of_k_curve,
    candidate_selection_audit,
    exact_nearest_squared_loss,
    illustrative_tables,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _synthetic_candidates(seed: int, conditions: int = 12, candidates: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tables = illustrative_tables()
    output = np.empty((conditions, candidates, 96), dtype=np.float32)
    for i in range(conditions):
        beam = tables.beam[i % len(tables.beam)]
        column = tables.column[i % len(tables.column)]
        exact = np.concatenate(
            [np.repeat(beam[j], 16) for j in range(4)]
            + [np.repeat(column[j], 16) for j in range(2)]
        ).astype(np.float32)
        noise_scale = np.linspace(0.22, 0.02, candidates, dtype=np.float32)
        table_range = np.concatenate(
            [np.repeat(np.ptp(tables.beam, axis=0)[j], 16) for j in range(4)]
            + [np.repeat(np.ptp(tables.column, axis=0)[j], 16) for j in range(2)]
        ).astype(np.float32)
        for k, scale in enumerate(noise_scale):
            output[i, k] = exact + rng.normal(0.0, scale, 96).astype(np.float32) * table_range
    return output


def smoke(seed: int = 2025) -> dict:
    """Run fast algorithmic checks without claiming model-performance reproduction."""

    set_seed(seed)
    tables = illustrative_tables()
    candidates = _synthetic_candidates(seed)
    first = candidates[:, 0]
    projected = tables.project(first)
    curve = best_of_k_curve(tables, candidates, ks=(1, 2, 4, 8, 16))
    result = {
        "profile": "algorithmic-smoke-test",
        "seed": seed,
        "reference_tables": "synthetic illustration; not official JIS data",
        "conditions": int(len(first)),
        "projection_distance_before_mean": float(tables.projection_distance(first).mean()),
        "projection_distance_after_mean": float(tables.projection_distance(projected).mean()),
        "projected_compliance_rate": float(tables.compliance(projected).mean()),
        "projection_idempotent": bool(np.array_equal(projected, tables.project(projected))),
        "best_of_k": {str(k): value for k, value in curve.items()},
    }
    result["selection_audit"] = candidate_selection_audit(tables, candidates)
    try:
        import torch

        legal = torch.tensor(projected[:1], dtype=torch.float32, requires_grad=True)
        legal_loss = exact_nearest_squared_loss(legal, tables)
        legal_loss.backward()
        result["legal_loss"] = float(legal_loss.item())
        result["legal_gradient_max_abs"] = float(legal.grad.abs().max().item())
    except ImportError:
        result["torch_check"] = "skipped: torch is not installed"
    return result


def load_private_dataset(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load only numeric arrays; never unpickle the recovered object-id field."""

    root = Path(dataset_dir)
    with np.load(root / "dataset_train.npz", allow_pickle=False) as archive:
        condition = np.asarray(archive["C"], dtype=np.float32)
        section = np.asarray(archive["X"], dtype=np.float32)
    metadata = json.loads((root / "dataset_meta.json").read_text(encoding="utf-8"))
    if section.ndim != 2 or section.shape[1] != 96:
        raise ValueError("dataset_train.npz must contain X with 96 columns")
    return condition, section, metadata


def _denormalize(x_norm, metadata: dict):
    import torch

    minimum = np.repeat(np.asarray(metadata["sbound_min"], dtype=np.float32), 16)
    maximum = np.repeat(np.asarray(metadata["sbound_max"], dtype=np.float32), 16)
    low = torch.as_tensor(minimum, dtype=x_norm.dtype, device=x_norm.device)
    high = torch.as_tensor(maximum, dtype=x_norm.dtype, device=x_norm.device)
    return x_norm * (high - low) + low


def train_recovered(
    dataset_dir: str | Path,
    beam_csv: str | Path,
    column_csv: str | Path,
    *,
    epochs: int,
    batch_size: int,
    hidden: int,
    n_critic: int,
    seed: int,
) -> dict:
    """Train a compact corrected conditional WGAN-GP on recovered numeric data."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .models import Critic, Generator, gradient_penalty

    set_seed(seed)
    condition_np, section_np, metadata = load_private_dataset(dataset_dir)
    tables = JISTables.from_csv(beam_csv, column_csv)
    condition = torch.from_numpy(condition_np)
    section = torch.from_numpy(section_np)
    loader = DataLoader(
        TensorDataset(condition, section),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim = 16
    generator = Generator(z_dim, condition.shape[1], hidden=hidden).to(device)
    critic = Critic(condition.shape[1], hidden=hidden).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.0, 0.9))
    history = []
    for epoch in range(1, epochs + 1):
        d_total = g_total = jis_total = 0.0
        steps = 0
        for batch_c, batch_x in loader:
            batch_c = batch_c.to(device)
            batch_x = batch_x.to(device)
            for _ in range(n_critic):
                z = torch.randn(len(batch_c), z_dim, device=device)
                fake = generator(z, batch_c).detach()
                penalty = gradient_penalty(critic, batch_x, fake, batch_c)
                loss_d = critic(fake, batch_c).mean() - critic(batch_x, batch_c).mean() + 10.0 * penalty
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
            z = torch.randn(len(batch_c), z_dim, device=device)
            fake = generator(z, batch_c)
            fake_mm = _denormalize(fake, metadata)
            loss_jis = exact_nearest_squared_loss(fake_mm, tables, normalized=True)
            loss_g = -critic(fake, batch_c).mean() + loss_jis
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            d_total += float(loss_d.item())
            g_total += float(loss_g.item())
            jis_total += float(loss_jis.item())
            steps += 1
        history.append(
            {
                "epoch": epoch,
                "critic_loss": d_total / steps,
                "generator_loss": g_total / steps,
                "exact_jis_loss": jis_total / steps,
            }
        )
    return {
        "profile": "corrected-recovered-data-training",
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden": hidden,
        "n_critic": n_critic,
        "samples": int(len(condition)),
        "history": history,
        "scope_warning": "A short run verifies execution only; it does not validate thesis performance claims.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "recovered"), default="smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--beam-csv", type=Path)
    parser.add_argument("--column-csv", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n-critic", type=int, default=1)
    args = parser.parse_args()
    if args.profile == "smoke":
        result = smoke(args.seed)
    else:
        missing = [name for name in ("dataset_dir", "beam_csv", "column_csv") if getattr(args, name) is None]
        if missing:
            parser.error("recovered profile requires --dataset-dir, --beam-csv, and --column-csv")
        result = train_recovered(
            args.dataset_dir,
            args.beam_csv,
            args.column_csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden=args.hidden,
            n_critic=args.n_critic,
            seed=args.seed,
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
