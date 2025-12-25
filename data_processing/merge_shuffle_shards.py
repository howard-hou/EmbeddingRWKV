# -*- coding: utf-8 -*-

import argparse
import gzip
import os
import random
import sys
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

def list_common_shards(input_dirs):
    """取所有目录共同拥有的 shard*.jsonl.gz（严格一一对应，按你的命名保留）。"""
    name2path = defaultdict(list)
    for d in input_dirs:
        for p in d.glob("shard*.jsonl.gz"):
            name2path[p.name].append(p)
    return name2path

def get_gz_lines(path: Path):
    """读取 gzip 文本，容错忽略坏编码。"""
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        return f.readlines()

def merge_shuffle_one(sub_shards, output_path, delete_input=False):
    """
    单个分片：
    1) 读取并合并
    2) 全量打乱
    3) 写出到目标 gzip
    4) 成功后删除输入分片（可选）
    """
    all_lines = []
    for sub in sub_shards:
        all_lines.extend(get_gz_lines(Path(sub)))

    if not all_lines:
        return ("skip", str(output_path))

    # Shuffle lines（保持你原先固定种子逻辑）
    random.seed(1337)
    random.shuffle(all_lines)

    # Write out to target gzip
    output_path = Path(output_path)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        f.writelines(all_lines)

    # Delete input shards
    if delete_input:
        for sub in sub_shards:
            try:
                os.remove(sub)
            except Exception as e:
                # 不阻断主流程，仅提示
                sys.stderr.write(f"[WARN] 删除失败 {sub}: {e}\n")

    return ("ok", str(output_path))

def main():
    ap = argparse.ArgumentParser(description="多进程合并+洗牌 shards（最小改动）")
    ap.add_argument("--inputs", nargs="+", required=True, help="输入目录（至少1个）")
    ap.add_argument("--output", required=True, help="输出目录")
    ap.add_argument("--delete_input", action="store_true", help="成功后删除输入分片")
    ap.add_argument("--workers", type=int, default=os.cpu_count() - 4, help="并发进程数")
    args = ap.parse_args()
    print('start with workers:', args.workers)

    input_dirs = [Path(p).resolve() for p in args.inputs]
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = list_common_shards(input_dirs)
    if not shards:
        sys.exit(1)

    # 任务打包（避免在主进程里读取大文件，只传轻量参数）
    tasks = []
    for s, paths in shards.items():
        sub_shards = [str(p) for p in paths]
        out_path = str(out_dir / s)
        tasks.append((sub_shards, out_path, args.delete_input))

    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(merge_shuffle_one, *t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Merging+Shuffling"):
            try:
                status, path = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    err += 1
            except Exception as e:
                err += 1
                sys.stderr.write(f"[ERR] 子任务失败: {e}\n")

    print(f"[SUMMARY] ok={ok}, skip={skip}, err={err}")

if __name__ == "__main__":
    main()
