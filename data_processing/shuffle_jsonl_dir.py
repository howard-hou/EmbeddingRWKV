#!/usr/bin/env python3
"""Shuffle and reshard JSONL.GZ files contained in a directory."""

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import List

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shuffle jsonl.gz files and write N shards")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing *.jsonl.gz files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where shuffled shards will be written",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        required=True,
        help="Number of shards to produce",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for shuffling",
    )
    return parser.parse_args()


def load_all_lines(input_dir: Path) -> List[str]:
    lines: List[str] = []
    for path in tqdm(sorted(input_dir.glob("*.jsonl.gz")), desc="read"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as reader:
            for raw_line in reader:
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                record["source"] = path.name
                lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    return lines


def write_shards(output_dir: Path, lines: List[str], num_shards: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(lines)
    for shard_idx in tqdm(range(num_shards), desc="write"):
        start = shard_idx * total // num_shards
        end = (shard_idx + 1) * total // num_shards
        shard_lines = lines[start:end]
        shard_path = output_dir / f"shard{shard_idx:05d}.jsonl.gz"
        with gzip.open(shard_path, "wt", encoding="utf-8") as writer:
            writer.writelines(shard_lines)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    lines = load_all_lines(input_dir)
    random.Random(args.seed).shuffle(lines)
    write_shards(output_dir, lines, args.num_shards)


if __name__ == "__main__":
    main()
