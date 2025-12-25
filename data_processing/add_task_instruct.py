#!/usr/bin/env python3
"""Augment JSONL.GZ records with task instructions."""

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict



def load_mapping(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping JSON to be an object, got {type(data)!r}")
    return data


def augment_file(mapping: Dict[str, str], input_path: Path, output_path: Path) -> None:
    with gzip.open(input_path, "rt", encoding="utf-8") as fin, gzip.open(
        output_path, "wt", encoding="utf-8"
    ) as fout:
        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "source" not in record:
                raise KeyError(f"Line {line_num}: missing 'source' field")
            source = record["source"]
            if source not in mapping:
                raise KeyError(
                    f"Line {line_num}: instruction for source '{source}' not found in mapping"
                )
            record["instruct"] = mapping[source]['instruction']
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add task instructions to a JSONL.GZ file using a source-to-instruction mapping."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input JSONL.GZ file.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path for the output JSONL.GZ file.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Path to the mapping JSON file (defaults to origneg-eng-v1/tools/origneg-eng-v1-task.json if present).",
    )
    args = parser.parse_args()
    if args.mapping is None:
        parser.error("--mapping is required because the default mapping file was not found.")
    return args


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    augment_file(mapping, args.input, args.output)


if __name__ == "__main__":
    main()
