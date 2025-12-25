"""Utility script to identify datasets that follow the Triplets format.

This script scans one or more `.jsonl.gz` files (or directories containing
such files) and prints the paths that contain triplet-formatted entries.

Usage:
    python detect_triplet_datasets.py path/to/dataset_dir
    python detect_triplet_datasets.py file1.jsonl.gz file2.jsonl.gz

The script inspects the JSON object on each line and classifies it according
to the dataset format documented for this project. Only datasets where every
inspected line is compatible with a triplet schema are reported.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable, Optional


TRIPLET_TYPES = {"Triplets", "Query-Triplets"}


def iter_jsonl_gz_files(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield all `.jsonl.gz` files contained in *paths*.

    If a directory is provided, it is scanned recursively.
    """

    for path in paths:
        if path.is_dir():
            yield from iter_jsonl_gz_files(sorted(path.rglob("*.jsonl.gz")))
        elif path.is_file() and path.suffixes[-2:] == [".jsonl", ".gz"]:
            yield path


def detect_example_type(example: object) -> Optional[str]:
    """Classify a parsed JSON example.

    Returns the dataset format name or ``None`` if the example could not be
    classified.
    """

    if isinstance(example, list):
        if len(example) == 3:
            return "Triplets"
        if len(example) == 2:
            return "Pairs"
    elif isinstance(example, dict):
        if {"query", "pos", "neg"}.issubset(example):
            if isinstance(example.get("pos"), list) and isinstance(
                example.get("neg"), list
            ):
                return "Query-Triplets"
        if {"set"} <= example.keys():
            return "Sets"
        if {"query", "pos"}.issubset(example):
            return "Query-Pairs"
    return None


def file_contains_only_triplets(path: Path, sample_limit: int = 1000) -> bool:
    """Return ``True`` if the file only contains triplet-style entries."""

    inspected = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                return False

            example_type = detect_example_type(example)
            if example_type not in TRIPLET_TYPES:
                return False

            inspected += 1
            if inspected >= sample_limit:
                break

    # Empty files are not considered triplet datasets.
    return inspected > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify datasets that are stored in the Triplets format."
    )
    parser.add_argument(
        "paths",
        metavar="PATH",
        nargs="+",
        type=Path,
        help="File or directory paths to inspect",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=1000,
        help="Maximum number of examples to inspect per file (default: 1000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_files = list(iter_jsonl_gz_files(args.paths))

    if not jsonl_files:
        print("No .jsonl.gz files found in the provided paths.")
        return

    for file_path in jsonl_files:
        if file_contains_only_triplets(file_path, sample_limit=args.sample_limit):
            print(file_path)


if __name__ == "__main__":
    main()
