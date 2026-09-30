#!/usr/bin/env python3
"""Fail closed on common secrets, private paths, notebook residue, and unsafe payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".html",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {".gitignore", "LICENSE"}
BANNED_BINARY_SUFFIXES = {".pkl", ".pickle", ".pt", ".pth", ".npz"}
APPROVED_PDF = Path("thesis/graduation-thesis-public-candidate.pdf")
APPROVED_PDF_SHA256 = "9948a2976b3c753a432e75c550ad0aea08d255b4db9f81983debf6a2ddfe5194"
DROP_NOTEBOOK_KEYS = {
    "authorship_tag",
    "base_uri",
    "executionInfo",
    "outputId",
    "referenced_widgets",
    "user",
    "widgets",
}
PATTERNS = {
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])"),
    "local-user-path": re.compile(r"/" + r"Users/|[A-Za-z]:\\" + r"Users\\"),
    "private-colab-mount": re.compile(r"/content/" + r"(?:drive|gdrive)"),
    "direct-drive-resource": re.compile(r"drive" + r"\.google\.com/(?:file/d/|drive/folders/|open\?id=|uc\?id=)"),
    "hugging-face-token": re.compile(r"h" + r"f_[A-Za-z0-9]{20,}"),
    "github-token": re.compile(r"g" + r"h(?:p|o|u|s|r)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"),
    "google-api-key": re.compile(r"A" + r"Iza[0-9A-Za-z_-]{30,}"),
    "openai-style-key": re.compile(r"s" + r"k-[A-Za-z0-9_-]{20,}"),
    "private-key-header": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "jwt": re.compile(r"e" + r"yJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    "bearer-token": re.compile(r"(?i)bearer\s+[A-Za-z0-9._~-]{20,}"),
    "wandb-key": re.compile(r"(?i)wandb(?:_api_key)?\s*[=:]\s*['\"]?[A-Za-z0-9]{20,}"),
    "aws-access-key": re.compile(r"A" + r"KIA[0-9A-Z]{16}"),
}


def load_forbidden(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def audit_notebook(path: Path, failures: list[str]) -> None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        failures.append(f"invalid notebook JSON: {path}")
        return
    residue = sorted(DROP_NOTEBOOK_KEYS.intersection(walk_keys(notebook)))
    if residue:
        failures.append(f"private notebook metadata keys in {path}: {', '.join(residue)}")
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            failures.append(f"executed notebook cell in {path}: cell {index}")


def audit_pdf(path: Path, relative: Path, failures: list[str]) -> None:
    if relative != APPROVED_PDF:
        failures.append(f"unapproved PDF payload: {relative}")
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != APPROVED_PDF_SHA256:
        failures.append(f"approved PDF hash mismatch: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--forbidden-file", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    forbidden = load_forbidden(args.forbidden_file)
    failures: list[str] = []
    scanned = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        relative = path.relative_to(root)
        if path.suffix.lower() in BANNED_BINARY_SUFFIXES:
            failures.append(f"unsafe/private binary payload: {relative}")
        if getattr(path.stat(), "st_blocks", 1) == 0 and path.stat().st_size > 0:
            failures.append(f"dataless file: {relative}")
        if path.suffix.lower() == ".pdf":
            audit_pdf(path, relative, failures)
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_FILENAMES:
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"non-UTF-8 text candidate: {relative}")
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{name}: {relative}")
        if any(value in text for value in forbidden):
            failures.append(f"caller-supplied private identifier: {relative}")
        if path.suffix.lower() == ".ipynb":
            audit_notebook(path, failures)
    if failures:
        print(f"AUDIT FAILED ({len(failures)} findings; values intentionally suppressed)")
        for finding in failures:
            print(f"- {finding}")
        return 1
    print(f"AUDIT PASSED: {scanned} text files; no configured privacy or credential signals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
