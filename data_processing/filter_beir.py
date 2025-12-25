import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, Iterator, List, Sequence

# English retrieval sources to keep (BEIR-aligned + other high-value corpora).
ALLOWED_SOURCES: frozenset[str] = frozenset(
    {
        "CodeFeedback-Filtered-Instruction.jsonl.gz",
        "ELI5-custom.jsonl.gz",
        "MEDI2BGE.jsonl.gz",
        "OpenOrca.jsonl.gz",
        "PubMedQA.jsonl.gz",
        "StackOverflowDupQuestions.jsonl.gz",
        "WikiAnswers_1107.jsonl.gz",
        "beir_trec_covid.jsonl.gz",
        "dbpedia-entity-generated-queries.jsonl.gz",
        "emb-triviaqa-train.jsonl.gz",
        "esci-product-search.jsonl.gz",
        "fever.jsonl.gz",
        "fiqa.jsonl.gz",
        "hotpotqa_fullwiki.jsonl.gz",
        "mmarco-eng.jsonl.gz",
        "msmarco-passage.jsonl.gz",
        "nfcorpus-generated-queries.jsonl.gz",
        "paq.jsonl.gz",
        "rag-dataset-12000.jsonl.gz",
        "raw_biorxiv.jsonl.gz",
        "raw_medrxiv.jsonl.gz",
        "reddit-title-body.jsonl.gz",
        "scidocs.jsonl.gz",
        "scifact.jsonl.gz",
        "squad_v2.jsonl.gz",
        "stackexchange_titlebody_best_and_down_voted_answer.jsonl.gz",
        "wikipedia-nq.jsonl.gz",
        "yahoo-answers.jsonl.gz",
    }
)

# BEIR mid-range sources to up-sample via pos/neg expansion.
MID_RANGE_SOURCES: frozenset[str] = frozenset(
    {
        "beir_trec_covid.jsonl.gz",
        "fever.jsonl.gz",
        "fiqa.jsonl.gz",
        "hotpotqa_fullwiki.jsonl.gz",
        "nfcorpus-generated-queries.jsonl.gz",
        "scidocs.jsonl.gz",
        "scifact.jsonl.gz",
        "wikipedia-nq.jsonl.gz",
    }
)

MIN_SAMPLES_PER_SOURCE = 1000


def validate_record(record: Dict) -> bool:
    """Return True if the record has non-empty pos/neg lists."""
    pos = record.get("pos")
    neg = record.get("neg")
    return isinstance(pos, list) and pos and isinstance(neg, list) and neg


def expand_mid_range_record(record: Dict, rng: random.Random) -> Iterator[Dict]:
    """Expand a mid-range record into max(len(pos), len(neg)) samples."""
    pos_list: Sequence = record.get("pos", [])
    neg_list: Sequence = record.get("neg", [])
    base = {k: v for k, v in record.items() if k not in {"pos", "neg"}}

    for pos in pos_list:
        expanded = dict(base)
        expanded["pos"] = [pos]
        expanded["neg"] = neg_list
        yield expanded


def process_file(
    input_path: str,
    output_path: str,
    *,
    seed: int | None,
) -> tuple[int, Counter, Counter]:
    rng = random.Random(seed) if seed is not None else random.Random()
    include_counts: Counter = Counter()
    skip_counts: Counter = Counter()
    per_source_lines: dict[str, List[str]] = defaultdict(list)

    with gzip.open(input_path, "rt", encoding="utf-8") as src:
        for raw_line in src:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            source = record.get("source")
            if source not in ALLOWED_SOURCES:
                skip_counts["source"] += 1
                continue
            if not validate_record(record):
                skip_counts["missing_pairs"] += 1
                continue
            include_counts[source] += 1
            iterable: Iterable[Dict] = (
                expand_mid_range_record(record, rng)
                if source in MID_RANGE_SOURCES
                else (record,)
            )
            for new_record in iterable:
                per_source_lines[source].append(json.dumps(new_record, ensure_ascii=False))

    total_written = 0
    output_lines: List[str] = []
    for source, lines in per_source_lines.items():
        if len(lines) < MIN_SAMPLES_PER_SOURCE and lines:
            needed = MIN_SAMPLES_PER_SOURCE - len(lines)
            lines.extend(rng.choices(lines, k=needed))
        output_lines.extend(lines)
        total_written += len(lines)

    rng.shuffle(output_lines)

    with gzip.open(output_path, "wt", encoding="utf-8") as dst:
        for line in output_lines:
            dst.write(f"{line}\n")

    return total_written, include_counts, skip_counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter shard data to English retrieval sources, expand BEIR mid-range "
            "records via asymmetric pairing, and ensure at least 1000 samples per source."
        )
    )
    parser.add_argument("input_path", help="Path to the source jsonl.gz file.")
    parser.add_argument("output_path", help="Where to write the filtered jsonl.gz file.")
    parser.add_argument(
        "--seed",
        type=int,
        default=222,
        help="Random seed used for pairing, shuffling, and per-source upsampling.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    written, include_counts, skip_counts = process_file(
        input_path=args.input_path,
        output_path=args.output_path,
        seed=args.seed,
    )

    kept = sum(include_counts.values())
    parser.exit(
        status=0,
        message=(
            f"Wrote {written} records to {args.output_path}\n"
            f"Kept {kept} records from {len(include_counts)} sources; "
            f"skipped {sum(skip_counts.values())} records.\n"
        ),
    )


if __name__ == "__main__":
    main()
