"""Utility script to evaluate a trained RWKV reranker."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from unittest import result
import torch
from pathlib import Path
import mteb
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from src.wrapper import CrossEncoderWrapper
from src.utils import parse_reranker_layer_idx
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,   # override any existing logging configuration
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a VisualRWKV reranker model")
    parser.add_argument("--model_path", required=True, help="Path to the trained VisualRWKV model checkpoint")
    parser.add_argument("--vision_tower_path", help="Path to the vision tower checkpoint used")
    parser.add_argument("--tokenizer_path", default="tokenizer/rwkv_vocab_v20230424.txt")
    parser.add_argument("--device", default="cuda", help="Device for evaluation (e.g., cuda or cpu)")
    parser.add_argument("--ctx_len", type=int, default=1024, help="Context length used during training")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of RWKV layers")
    parser.add_argument("--n_embd", type=int, default=512, help="Embedding dimension of the RWKV model")
    parser.add_argument("--vocab_size", default=65536, type=int)
    parser.add_argument("--dim_att", type=int, default=0, help="Attention dimension (defaults to --n_embd)")
    parser.add_argument("--dim_ffn", type=int, default=0)
    parser.add_argument("--head_size_a", type=int, default=64, help="Attention head size")
    parser.add_argument("--head_size_divisor", type=int, default=8, help="Group norm head size divisor")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--grad_cp", type=int, default=0, help="Enable gradient checkpointing (1 to enable)")
    parser.add_argument("--num_token_per_image", type=int, default=256, help="Number of visual tokens per image")
    parser.add_argument(
        "--reranker_layer_idx",
        type=str,
        default="0",
        help="Comma-separated RWKV layer indices used by the reranker (e.g. 0,5,11).",
    )
    parser.add_argument("--use_shared_state", action="store_true", help="Reuse the final RWKV state for every reranker layer (keeps RWKV depth).")
    parser.add_argument("--load_model",default="", help="Path to the base RWKV checkpoint")
    parser.add_argument("--precision",default="bf16", help="model3 only")
    parser.add_argument(
        "--benchmark_name",
        help="Name of the MTEB benchmark to evaluate.",
    )
    parser.add_argument(
        "--task_name",
        help="Name of the individual MTEB task to evaluate when no benchmark is specified.",
    )
    parser.add_argument(
        "--previous_results_dir",
        type=Path,
        default=Path("results_recall"),
        help="Directory containing first-stage retrieval results for reranking.",
    )
    args = parser.parse_args()
    args.reranker_layer_idx = parse_reranker_layer_idx(args.reranker_layer_idx)
    return args


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_JIT_ON"] = "0"

    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)


def build_model(args: argparse.Namespace):
    from src.model import RWKVReRanker
    tokenizer = TRIE_TOKENIZER(args.tokenizer_path)
    args.MODEL_NAME = args.load_model.strip(".pth")

    model = RWKVReRanker(args)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    load_msg = model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded model from {args.model_path}: {load_msg}")
    model = model.bfloat16().to(args.device)
    model.eval()

    return model, tokenizer


def find_previous_results_file(task_name: str, search_dir: Path) -> Path:
    direct_match = search_dir / f"{task_name}_default_predictions.json"
    if direct_match.exists():
        return direct_match
    logging.warning(f"No direct match found for {task_name} in {search_dir}.")
    return None


def filter_tasks_by_task_types(tasks, task_types):
    _task_types = set(task_types)
    return [t for t in tasks if t.metadata.type in _task_types]


def main() -> None:
    args = parse_args()
    configure_environment(args)
    model, tokenizer = build_model(args)
    template = "Instruct: Given a query, retrieve documents that answer the query\nDocument: {document}\nQuery: {query}"

    wrapper = CrossEncoderWrapper(model, tokenizer, template)
    output_dir = Path(args.model_path).parent / "eval_results" / Path(args.model_path).stem
    tasks = mteb.get_benchmark(args.benchmark_name).tasks if args.benchmark_name else mteb.get_tasks(tasks=[args.task_name])
    tasks = filter_tasks_by_task_types(tasks, ["Retrieval"]) # only retrieval tasks
    # start evaluation
    task_scores: list[tuple[str, float]] = []
    for task in tasks:
        task_name = task.metadata_dict["name"]
        previous_results_path = find_previous_results_file(
            task_name, args.previous_results_dir
        )
        evaluation = mteb.MTEB(tasks=[task])
        task_result = evaluation.run(
            wrapper,
            top_k=100,
            save_predictions=True,
            output_folder=output_dir,
            previous_results=str(previous_results_path),
            verbose=1,
        )[0]

        score = float(task_result.get_score())
        logging.info(f"{task_name}: {round(score * 100, 2)}")
        task_scores.append((task_name, score))
    # log overall results
    avg_score = sum(score for _, score in task_scores) / len(task_scores)
    task_scores.append(("Mean (Task)", avg_score))
    if args.benchmark_name:
        logging.info(f"Average score over benchmark {args.benchmark_name}: {round(avg_score * 100, 2)}")
    else:
        logging.info(f"Score for task {args.task_name}: {round(avg_score * 100, 2)}")
    # save aggregated results
    from tabulate import tabulate
    table_str = tabulate(
        task_scores,
        headers=["task_name", "main_score"],
        tablefmt="github",
    )
    filename = f"{args.benchmark_name}_table.txt" if args.benchmark_name else f"{args.task_name}_table.txt"
    txt_path = output_dir / filename
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")


if __name__ == "__main__":
    main()
