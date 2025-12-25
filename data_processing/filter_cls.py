import argparse
import gzip
import json
from typing import Optional


def filter_retr_tasks(input_path: str, output_path: str) -> int:
    """Filter records whose `task` field not equals "[CLS]" from a jsonl.gz file."""
    matched = 0
    with gzip.open(input_path, 'rt', encoding='utf-8') as src, gzip.open(output_path, 'wt', encoding='utf-8') as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('task') != '[CLS]':
                if line.endswith('\n'):
                    dst.write(line)
                else:
                    dst.write(f"{line}\n")
                matched += 1
    return matched


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Extract records with task != "[CLS]" from a jsonl.gz file.')
    parser.add_argument('input_path', help='Path to the source jsonl.gz file.')
    parser.add_argument('output_path', help='Where to write the filtered jsonl.gz file.')
    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)
    matched = filter_retr_tasks(parsed.input_path, parsed.output_path)
    parser.exit(status=0, message=f'Wrote {matched} [RETR] records to {parsed.output_path}\n')


if __name__ == '__main__':
    main()
