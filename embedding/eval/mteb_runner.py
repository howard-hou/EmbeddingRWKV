import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import mteb
import numpy as np

from custom_embedding_model import VisualRWKVMTEBConfig, VisualRWKVMTEBModel
from custom_late_interaction_model import CustomLateInteractionModel
from custom_reranker_model import CustomRerankerModel, RWKVRerankerMTEBConfig
from src.task_tag import get_task_tag_by_task
from src.task_instruct import get_task_instruct_by_task

from tabulate import tabulate
import warnings
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples.*"
)



AVAILABLE_BENCHMARKS = {
  "BEIR": "BEIR",
  "BEIR_NL": "BEIR-NL",
  "BRIGHT": "BRIGHT",
  "BRIGHT_LONG": "BRIGHT (long)",
  "BUILTBENCH_ENG": "BuiltBench(eng)",
  "CHEMTEB": "ChemTEB",
  "COIR": "CoIR",
  "CODERAG": "CodeRAG",
  "ENCODECHKA": "Encodechka",
  "FOLLOWIR": "FollowIR",
  "JINAVDR": "JinaVDR",
  "LONGEMBED": "LongEmbed",
  "MIEB_IMG": "MIEB(Img)",
  "MIEB_MULTILINGUAL": "MIEB(Multilingual)",
  "MIEB_ENG": "MIEB(eng)",
  "MIEB_LITE": "MIEB(lite)",
  "MINERSBITEXTMINING": "MINERSBitextMining",
  "MTEB_CODE_V1": "MTEB(Code, v1)",
  "MTEB_EUROPE_V1": "MTEB(Europe, v1)",
  "MTEB_INDIC_V1": "MTEB(Indic, v1)",
  "MTEB_LAW_V1": "MTEB(Law, v1)",
  "MTEB_MEDICAL_V1": "MTEB(Medical, v1)",
  "MTEB_MULTILINGUAL_V1": "MTEB(Multilingual, v1)",
  "MTEB_MULTILINGUAL_V2": "MTEB(Multilingual, v2)",
  "MTEB_SCANDINAVIAN_V1": "MTEB(Scandinavian, v1)",
  "MTEB_CMN_V1": "MTEB(cmn, v1)",
  "MTEB_DEU_V1": "MTEB(deu, v1)",
  "MTEB_ENG_V1": "MTEB(eng, v1)",
  "MTEB_ENG_V2": "MTEB(eng, v2)",
  "MTEB_FAS_V1": "MTEB(fas, v1)",
  "MTEB_FRA_V1": "MTEB(fra, v1)",
  "MTEB_JPN_V1": "MTEB(jpn, v1)",
  "MTEB_KOR_V1": "MTEB(kor, v1)",
  "MTEB_POL_V1": "MTEB(pol, v1)",
  "MTEB_RUS_V1": "MTEB(rus, v1)",
  "NANOBEIR": "NanoBEIR",
  "R2MED": "R2MED",
  "RAR_B": "RAR-b",
  "RUSCIBENCH": "RuSciBench",
  "VN_MTEB_VIE_V1": "VN-MTEB (vie, v1)",
  "VIDORE_V1": "ViDoRe(v1)",
  "VIDORE_V2": "ViDoRe(v2)",
  "VISUALDOCUMENTRETRIEVAL": "VisualDocumentRetrieval"
}

_TASK_TYPE = (
    "BitextMining",
    "Classification",
    "Clustering",
    "InstructionRetrieval",
    "MultilabelClassification",
    "PairClassification",
    "Regression",
    "Reranking",
    "Retrieval",
    "Speed",
    "STS",
    "Summarization",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MTEB evaluation for RWKV embeddings."
    )
    parser.add_argument("--model-path", help="Path to model checkpoint")
    parser.add_argument("--vision-tower-path", help="Path to vision tower")
    parser.add_argument("--rwkv-path", help="Path to RWKV model checkpoint")
    parser.add_argument("--reranker-path", help="Path to reranker model checkpoint")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--benchmark-name", help="Name of the benchmark to run")
    target_group.add_argument("--task-name", help="Name of a single task to run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="results_mteb", help="Directory for outputs")
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--vocab_size", type=int, default=65536)
    parser.add_argument("--n-embd", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples for quick testing")
    parser.add_argument("--eos-chunk-size", type=int, default=512, help="Chunk size for multiple EOS token")
    parser.add_argument('--use_instruct', action="store_true")
    parser.add_argument("--late-interaction", action="store_true", help="Use late interaction model")
    parser.add_argument("--state-reranking", action="store_true", help="Use reranker model")
    parser.add_argument("--state-reranker-dim", type=int, default=256)
    parser.add_argument("--state-reranker-layers", type=int, default=3)
    parser.add_argument("--task-types",choices=_TASK_TYPE, nargs="+", help="Filter benchmark tasks by task type(s).")
    return parser.parse_args()


def filter_tasks_by_task_types(tasks, task_types):
    _task_types = set(task_types)
    return [t for t in tasks if t.metadata.type in _task_types]

def get_cfg(args):
    if args.state_reranking:
        cfg = RWKVRerankerMTEBConfig(
            rwkv_path=args.rwkv_path,
            reranker_path=args.reranker_path,
            ctx_len=args.ctx_len,
            n_layer=args.n_layer,
            vocab_size=args.vocab_size,
            n_embd=args.n_embd,
            eos_chunk_size=args.eos_chunk_size,
            state_reranker_dim=args.state_reranker_dim,
            state_reranker_layers=args.state_reranker_layers,
        )
    else:
        cfg = VisualRWKVMTEBConfig(
            model_path=args.model_path,
            vision_tower_path=args.vision_tower_path,
            ctx_len=args.ctx_len,
            vocab_size=args.vocab_size,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            eos_chunk_size=args.eos_chunk_size,
        )
    return cfg

def main():
    args = parse_args()

    cfg = get_cfg(args)
    # initialize model
    if args.late_interaction:
        print("Using CustomLateInteractionModel for evaluation.")
        model = CustomLateInteractionModel(cfg, device=args.device)
    elif args.state_reranking:
        print("Using CustomRerankerModel for evaluation.")
        model = CustomRerankerModel(cfg, device=args.device)
    else:
        print("Using VisualRWKVMTEBModel for evaluation.")
        model = VisualRWKVMTEBModel(cfg, device=args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    # get tasks
    tasks: List = []
    run_identifier: str
    if args.benchmark_name:
        real_benchmark_name = AVAILABLE_BENCHMARKS.get(
            args.benchmark_name, args.benchmark_name
        )
        if real_benchmark_name is None:
            raise ValueError(f"Unknown benchmark name: {args.benchmark_name}")
        print(f"Running MTEB benchmark: {real_benchmark_name}")
        benchmark = mteb.get_benchmark(real_benchmark_name)
        tasks = benchmark.tasks
        run_identifier = args.benchmark_name
        if args.late_interaction:
            run_identifier += "-late"
    else:
        print(f"Running single task: {args.task_name}")
        tasks = mteb.get_tasks(tasks=[args.task_name])
        if not tasks:
            raise ValueError(f"Unknown task name: {args.task_name}")
        run_identifier = args.task_name
        if args.late_interaction:
            run_identifier += "-late"
    # filter tasks by type if specified
    if args.task_types is not None:
        filtered_tasks = filter_tasks_by_task_types(tasks, args.task_types)
        if not filtered_tasks:
            available = sorted({getattr(task, "task_type", None) for task in tasks})
            raise ValueError(
                "No tasks remaining after filtering. Requested types "
                f"{sorted(args.task_types)}; available types: {available}"
            )
        print(
            "Filtering tasks by task types: "
            + ", ".join(sorted(args.task_types))
        )
        tasks = filtered_tasks
    else:
        print("No task type filter specified; evaluating all available task types.")

    # set tag for each task
    for i, task in enumerate(tasks):
        tag = get_task_tag_by_task(task)
        model.task2tag[task.metadata.name] = tag
    # set instruction for each task if use_instruct is True
    if args.use_instruct:
        for i, task in enumerate(tasks):
            instruction = get_task_instruct_by_task(task)
            model.task2instruct[task.metadata.name] = instruction
            print(f"Task {i}: {task.metadata.name}, Instruction: {instruction}")

    evaluation = mteb.MTEB(tasks=tasks)

    if args.model_path is not None:
        model_name = Path(args.model_path).parent.name + "." + Path(args.model_path).stem
    else:
        model_name = Path(args.reranker_path).parent.name + "." + Path(args.reranker_path).stem
    output_dir = Path(args.output_dir) / f"{model_name}.{run_identifier}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluation.run(
        model,
        encode_kwargs={"batch_size": args.batch_size},
        output_folder=output_dir,
        overwrite_results=False,
        limit=args.limit if hasattr(args, "limit") else None,
    )

    task_scores: List[Tuple[str, float]] = []
    type_scores: defaultdict[str, List[float]] = defaultdict(list)
    for result in results:
        score = float(result.get_score())
        task_scores.append((result.task_name, score))
        type_scores[result.task_type].append(score)

    aggregated_sections: List[Tuple[str, float]] = []
    if task_scores:
        mean_task = float(np.mean([score for _, score in task_scores]))
        aggregated_sections.append(("Mean (Task)", mean_task))

    type_means: List[Tuple[str, float]] = []
    for task_type, scores in sorted(type_scores.items()):
        type_means.append((task_type, float(np.mean(scores))))

    if type_means:
        mean_task_type = float(np.mean([score for _, score in type_means]))
        type_means_with_avg = [("Mean (Task Type)", mean_task_type)] + type_means
        aggregated_sections.extend(type_means_with_avg)

    table_str = tabulate(
        aggregated_sections,
        headers=["task_name", "main_score"],
        tablefmt="github",
    )

    print(table_str)
    txt_path = output_dir / "aggregated_table.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

if __name__ == "__main__":
    main()