"""Restore source files from JSONL(.gz) datasets.

This utility walks over all ``*.jsonl`` or ``*.jsonl.gz`` files within an
input directory, parses each record, and writes the value of a configured text
field back into files referenced by the ``source`` field.  It is useful when a
JSONL dataset was produced by chunking files and storing metadata about the
originating file in each record.

The script keeps a limited number of output file handles open at once to avoid
running into ``EMFILE`` errors on large datasets.  By default the ``text`` field
is written verbatim to the restored files; this can be customised via the
``--text-field`` argument.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator


class FileHandleCache:
    """LRU cache for output file handles."""

    def __init__(self, max_open_files: int, encoding: str) -> None:
        if max_open_files < 1:
            raise ValueError("max_open_files must be at least 1")
        self._max_open_files = max_open_files
        self._encoding = encoding
        self._handles: "OrderedDict[Path, io.TextIOBase]" = OrderedDict()

    def get(self, path: Path) -> io.TextIOBase:
        handle = self._handles.get(path)
        if handle is not None:
            self._handles.move_to_end(path)
            return handle

        if len(self._handles) >= self._max_open_files:
            _, oldest_handle = self._handles.popitem(last=False)
            oldest_handle.close()

        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding=self._encoding)
        self._handles[path] = handle
        return handle

    def close_all(self) -> None:
        while self._handles:
            _, handle = self._handles.popitem(last=False)
            handle.close()


def iter_jsonl_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, line)`` pairs from a JSONL or JSONL.GZ file."""

    if path.suffix == ".gz":
        open_fn = lambda: gzip.open(path, mode="rt", encoding="utf-8")
    elif path.suffix == ".jsonl":
        open_fn = lambda: path.open("r", encoding="utf-8")
    else:
        return

    with open_fn() as fh:  # type: ignore[call-arg]
        for idx, line in enumerate(fh, start=1):
            yield idx, line.rstrip("\n")


def find_jsonl_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.jsonl")):
        yield path
    for path in sorted(root.rglob("*.jsonl.gz")):
        yield path


def restore_sources(
    input_dir: Path,
    output_dir: Path,
    text_field: str,
    max_open_files: int,
    encoding: str,
) -> None:
    cache = FileHandleCache(max_open_files=max_open_files, encoding=encoding)
    total_records = 0
    write_records = 0

    try:
        for jsonl_path in find_jsonl_files(input_dir):
            for line_no, raw_line in iter_jsonl_lines(jsonl_path):
                line = raw_line.strip()
                if not line:
                    continue
                total_records += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"Warning: failed to parse JSON in {jsonl_path} line {line_no}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                source = record.get("source")
                if not source:
                    skipped_no_source += 1
                    continue
                if source.endswith(".gz"):
                    source = source[: -len(".gz")]

                output_path = output_dir / source
                handle = cache.get(output_path)
                handle.write(line + "\n")
                write_records += 1
    finally:
        cache.close_all()

    print(
        "Processed records: {processed}, written records: {written}".format(
            processed=total_records,
            written=write_records,
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing JSONL or JSONL.GZ files")
    parser.add_argument("output_dir", type=Path, help="Destination directory for restored sources")
    parser.add_argument(
        "--text-field",
        default="text",
        help="JSON field containing the content to write back to the source files (default: text)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding used for the restored files (default: utf-8)",
    )
    parser.add_argument(
        "--max-open-files",
        type=int,
        default=128,
        help="Maximum number of output files to keep open simultaneously (default: 128)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    restore_sources(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        text_field=args.text_field,
        max_open_files=args.max_open_files,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
