#!/usr/bin/env python3
"""Verify the recovered legacy generator without unsafe pickle object loading.

This script needs private, non-redistributed inputs.  It reports both a
deployable target-independent JIS projection-distance selector and the legacy
hidden-target oracle diagnostic.  The latter is never presented as an
inference-time metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gan_jis import JISTables


class LegacyGenerator(nn.Module):
    """Architecture recorded in the recovered strict_v2 package."""

    def __init__(self, z_dim: int = 64, c_dim: int = 5, x_dim: int = 96):
        super().__init__()
        layers: list[nn.Module] = []
        width = z_dim + c_dim
        for hidden in (256, 256, 256):
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.extend([nn.Linear(width, x_dim), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([noise, condition], dim=1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--beam-csv", required=True, type=Path)
    parser.add_argument("--column-csv", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.dataset_dir / "dataset_test.npz", allow_pickle=False) as archive:
        condition = np.asarray(archive["C"], dtype=np.float32)
        target = np.asarray(archive["X"], dtype=np.float32)
    metadata = json.loads((args.dataset_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    low = np.repeat(np.asarray(metadata["sbound_min"], dtype=np.float32), 16)
    high = np.repeat(np.asarray(metadata["sbound_max"], dtype=np.float32), 16)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    generator = LegacyGenerator()
    generator.load_state_dict(payload["G"])
    generator.eval()
    # Seed after module construction, whose random initialization consumes RNG.
    torch.manual_seed(args.seed)
    c_tensor = torch.from_numpy(condition)
    samples = []
    with torch.no_grad():
        for _ in range(args.k):
            noise = torch.randn(len(condition), 64)
            samples.append(generator(noise, c_tensor).numpy())
    candidates_norm = np.stack(samples, axis=1)
    candidates_mm = candidates_norm * (high - low) + low
    tables = JISTables.from_csv(args.beam_csv, args.column_csv)
    raw_distance = tables.projection_distance(
        candidates_mm.reshape(-1, 96), normalized=False
    ).reshape(len(condition), args.k)
    raw_l2 = raw_distance * np.sqrt(32.0)
    oracle_l1 = np.abs(candidates_norm - target[:, None, :]).mean(axis=2)
    result = {
        "profile": "recovered-private-checkpoint-verification",
        "checkpoint_sha256": sha256(args.checkpoint),
        "seed": args.seed,
        "test_conditions": int(len(condition)),
        "candidates_per_condition": args.k,
        "deployable_selection": {
            "rule": "minimum raw-mm complete-tuple projection distance; target-independent",
            "top1_raw_tuple_l2_mean": float(raw_l2[:, 0].mean()),
            "best_of_k_raw_tuple_l2_mean": float(raw_l2.min(axis=1).mean()),
            "top1_raw_tuple_rms_mean": float(raw_distance[:, 0].mean()),
            "best_of_k_raw_tuple_rms_mean": float(raw_distance.min(axis=1).mean()),
        },
        "oracle_diagnostic": {
            "best_of_k_normalized_l1_mean": float(oracle_l1.min(axis=1).mean()),
            "warning": "uses hidden target X and cannot be selected at inference",
        },
        "security": "checkpoint loaded with weights_only=True; NPZ object ids were not loaded",
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
