#!/usr/bin/env python3
"""Distribute lines from jsonl.gz files into sharded outputs.

Usage:
    python data_processing/distribute_jsonl_dir.py \
        --input_dir data/input \
        --output_dir data/output \
        --num_shards 400
"""
import argparse
import gzip
import os
import random
from pathlib import Path
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distribute JSONL.GZ files into shard files.")
    parser.add_argument("--input_dir", required=True, help="Directory containing *.jsonl.gz files")
    parser.add_argument("--output_dir", required=True, help="Directory where shards will be written")
    parser.add_argument("--num_shards", type=int, default=400, help="Number of output shards")
    parser.add_argument("--shuffle_shards", action="store_true", help="shuffle shard")
    return parser.parse_args()


def open_shards(output_dir: Path, num_shards: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = []
    for i in tqdm(range(num_shards), desc='open'):
        shard_path = output_dir / f"shard{i:03d}.jsonl.gz"
        writers.append(gzip.open(shard_path, "wt", encoding="utf-8"))
    return writers


def process_inputs(input_dir: Path, writers):
    shard_idx = 0
    num_shards = len(writers)
    for path in tqdm(sorted(input_dir.glob("*.jsonl.gz")), desc='process'):
        with gzip.open(path, "rt", encoding="utf-8") as reader:
            for line in reader:
                writers[shard_idx].write(line)
                shard_idx = (shard_idx + 1) % num_shards
        path.unlink()


def close_shards(writers):
    print("close shards start")
    for w in writers:
        w.close()


def shuffle_shards(output_dir: Path):
    for path in tqdm(sorted(output_dir.glob("shard*.jsonl.gz")), desc='shuffle'):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = f.readlines()
        random.shuffle(lines)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.writelines(lines)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    writers = open_shards(output_dir, args.num_shards)
    try:
        process_inputs(input_dir, writers)
    finally:
        close_shards(writers)
    if args.shuffle_shards:
        shuffle_shards(output_dir)


if __name__ == "__main__":
    main()
