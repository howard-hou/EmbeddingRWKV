"""Command-line tool to generate a sampling configuration from gzipped JSONL files.

The script walks over an input directory, counts the number of JSONL records in
each ``*.jsonl.gz`` file and proposes a sampling ratio for each file. Smaller
datasets are kept in full while larger datasets are downsampled so their
effective number of samples is comparable to the median sized dataset. The
result is written as a structured JSON document that is easy to tweak manually
*and* straightforward for other programs to load.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import statistics
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm


@dataclass
class DatasetStats:
    path: Path
    line_count: int
    sample_ratio: float

    @property
    def sampled_lines(self) -> int:
        return int(self.line_count * self.sample_ratio)

    def to_config_entry(self, share: float) -> dict[str, float | int | str]:
        """Return a JSON-serialisable representation of the dataset."""

        return {
            "file": self.path.name,
            "total_lines": self.line_count,
            "sample_ratio": round(self.sample_ratio, 3),
            "sampled_lines": self.sampled_lines,
            "dataset_share": round(share, 3),
        }


def count_lines(path: Path) -> int:
    """Count the number of lines in a ``.jsonl.gz`` file."""

    count = 0
    with gzip.open(path, mode="rt", encoding="utf-8") as fh:
        for _ in fh:
            count += 1
    return count


def compute_sampling_ratio(line_count: int, target_lines: float) -> float:
    """Return a sampling ratio that does not exceed 1.0.

    ``target_lines`` is the desired number of effective lines for large
    datasets.  Smaller datasets will use the full dataset (ratio of 1.0).
    """

    if line_count == 0:
        return 0.0
    ratio = target_lines / line_count
    return min(1.0, ratio)


def generate_config(
    input_dir: Path, target_lines: float | None
) -> tuple[list[DatasetStats], float, float]:
    """Gather statistics for all ``*.jsonl.gz`` files in ``input_dir``."""

    files = sorted(input_dir.glob("*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(
            f"No .jsonl.gz files found in directory: {input_dir}"
        )

    line_counts = [count_lines(path) for path in tqdm(files)]

    if target_lines is None:
        target_lines = int(statistics.median(line_counts))

    datasets: list[DatasetStats] = []
    for path, lines in zip(files, line_counts):
        ratio = compute_sampling_ratio(lines, target_lines)
        datasets.append(DatasetStats(path=path, line_count=lines, sample_ratio=ratio))

    total_sampled = sum(ds.sampled_lines for ds in datasets)
    return datasets, total_sampled


def format_config(
    datasets: list[DatasetStats],
    total_sampled: float,
) -> str:
    """Render the sampling configuration as CSV data with a header row."""

    output = io.StringIO()
    fieldnames = [
        "file",
        "total_lines",
        "sample_ratio",
        "sampled_lines",
        "dataset_share",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for stats in datasets:
        share = 0.0 if total_sampled == 0 else stats.sampled_lines / total_sampled
        row = stats.to_config_entry(share=share)
        writer.writerow(row)

    return output.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sampling configuration based on the number of lines in "
            "JSONL.GZ datasets."
        )
    )
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing *.jsonl.gz files"
    )
    parser.add_argument(
        "output_file", type=Path, help="Path to the sampling configuration file"
    )
    parser.add_argument(
        "--target-lines",
        type=float,
        default=None,
        help=(
            "Desired effective number of lines for each dataset. When omitted, "
            "the median line count is used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    target_lines: float | None = args.target_lines

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    datasets, total_sampled = generate_config(input_dir, target_lines)

    output_text = format_config(
        datasets=datasets,
        total_sampled=total_sampled,
    )

    args.output_file.write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
