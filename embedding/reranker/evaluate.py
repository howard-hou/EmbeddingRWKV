"""Utility script to evaluate a trained RWKV reranker."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import torch
from pathlib import Path

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from src.wrapper import CrossEncoderWrapper
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
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
    parser.add_argument("--device", default="cuda:0", help="Device for evaluation (e.g., cuda:0 or cpu)")
    parser.add_argument("--ctx_len", type=int, default=1024, help="Context length used during training")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of RWKV layers")
    parser.add_argument("--reranker_layer_idx", type=str, default="0", help="Comma-separated RWKV layer indices used by the reranker (e.g. 0,5,11).")
    parser.add_argument("--use_shared_state", action="store_true", help="Reuse the final RWKV state for each reranker layer while keeping the original depth.")
    parser.add_argument("--n_embd", type=int, default=512, help="Embedding dimension of the RWKV model")
    parser.add_argument("--vocab_size", default=65536, type=int)
    parser.add_argument("--dim_att", type=int, default=0, help="Attention dimension (defaults to --n_embd)")
    parser.add_argument("--dim_ffn", type=int, default=0)
    parser.add_argument("--head_size_a", type=int, default=64, help="Attention head size")
    parser.add_argument("--head_size_divisor", type=int, default=8, help="Group norm head size divisor")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--grad_cp", type=int, default=0, help="Enable gradient checkpointing (1 to enable)")
    parser.add_argument("--num_token_per_image", type=int, default=256, help="Number of visual tokens per image")
    parser.add_argument("--load_model",default="", help="Path to the base RWKV checkpoint")
    parser.add_argument("--precision",default="bf16", help="model3 only")
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


def main() -> None:
    args = parse_args()
    configure_environment(args)
    model, tokenizer = build_model3(args)
    template = "Instruct: Given a query, retrieve documents that answer the query\nDocument: {document}\nQuery: {query}"

    wrapper = CrossEncoderWrapper(model, tokenizer, template)
    evaluator = CrossEncoderNanoBEIREvaluator()
    output_path = Path(args.model_path).parent / "eval_results" / Path(args.model_path).stem
    with torch.inference_mode():
        results = evaluator(wrapper, output_path=output_path)
    logging.info(f"{evaluator.primary_metric}: {round(results[evaluator.primary_metric] * 100, 2)}")


if __name__ == "__main__":
    main()
