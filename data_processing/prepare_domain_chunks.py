#!/usr/bin/env python3
import json
import gzip
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Any

def stratified_round_robin_grouping(samples: List[Dict[str, Any]], chunk_size: int = 64):
    """
    Round-robin stratified grouping to ensure each group contains a balanced mix of different labels 
    without repetition within the group.
    Updated to support pos_image as label key if pos text is missing.
    """
    # 1. label buckets
    label_buckets = defaultdict(list)
    for s in samples:
        # 尝试获取标签 key，优先取 pos 文本，其次取 pos_image，如果都没有则跳过或归为 unknowns
        key = None
        if s.get("pos"):
            key = s["pos"][0]
        elif s.get("pos_image"):
            key = s["pos_image"][0]
        
        if key:
            label_buckets[key].append(s)

    if not label_buckets:
        return []

    # 2. get sample_len
    if len(label_buckets) == 2 or len(label_buckets) == 3:
        sample_len = min(len(v) for v in label_buckets.values())
    else:
        sample_len = np.percentile([len(v) for v in label_buckets.values()], 80).astype(int)

    # 3. round-robin assembly
    groups = []
    for i in range(sample_len):
        group = []
        for lbl in label_buckets:
            if i < len(label_buckets[lbl]):
                group.append(label_buckets[lbl][i])
        if len(group) >= 2:
            for j in range(0, len(group), chunk_size):
                sub_group = group[j:j+chunk_size]
                groups.append(sub_group)

    return groups

def load_jsonl_gz(path):
    data = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def detect_domain_type(data_by_domain, label_threshold=200):
    """根据 pos 的多样性判断 domain 类型"""
    domain_type = {}
    for domain, samples in data_by_domain.items():
        pos_texts = []
        for s in samples:
            pos_texts.extend(s.get("pos", []))
            pos_texts.extend(s.get("pos_image", []))
        pos_counter = Counter(pos_texts)
        unique_pos = len(pos_counter)
        if unique_pos == 0:
            # 防止除以零，如果没有 pos 信息，视为 general
            domain_type[domain] = "general"
            print(f"Domain '{domain}': {len(samples)} samples, 0 unique pos -> general")
            continue
            
        sample_unique_ratio = len(samples) / unique_pos
        if unique_pos <= label_threshold and sample_unique_ratio >= 5:
            domain_type[domain] = "classification"
        else:
            domain_type[domain] = "general"
        print(f"Domain '{domain}': {len(samples)} samples, {unique_pos} unique pos, sample/unique={sample_unique_ratio:.2f} -> {domain_type[domain]}")
    return domain_type

def chunkify(samples, chunk_size, n_buffer=3):
    """
    按 chunk_size 切分 samples，同时确保：
    - 同一个 chunk 内 query (text & image) 不重复
    - pos (text & image) 不重复 (防止 in-batch false positives)
    - neg 不与 query/pos overlap（会被过滤掉）
    - 保留 n_buffer×chunk_size 的缓冲区以维持平滑切分
    """
    buffer = []
    i = 0
    n = len(samples)

    while i < n or buffer:
        # 填充缓冲区
        while len(buffer) < n_buffer * chunk_size and i < n:
            buffer.append(samples[i])
            i += 1

        clean_chunk = []
        
        # 记录当前 chunk 内已存在的 query 和 pos，用于防撞
        seen_q = set()          # Query Text
        seen_q_img = set()      # Query Image
        seen_p = set()          # Pos Text
        seen_p_img = set()      # Pos Image

        next_buffer = []

        for j, s in enumerate(buffer):
            # 获取当前样本的字段
            q = s.get("query", "")
            q_img = s.get("query_image", "")
            
            pos = set(s.get("pos", []))
            pos_img = set(s.get("pos_image", []))
            
            neg = set(s.get("neg", []))
            # 注意：如果 neg 也有 image，逻辑类似，但通常 neg image 不做强过滤，这里保持原逻辑仅过滤 text neg
            # 如果需要过滤 neg image，请同样处理 neg = {n for n in neg_img ...}

            # 过滤掉与 query 或 pos 重叠的 neg (Self-Consistency)
            neg = {n for n in neg if n not in pos and n != q}

            # --- 冲突检测 (Collision Detection) ---
            
            # 1. 检查 Query 冲突 (同一张图或同一句话作为 Query 出现过)
            # 注意：空字符串/None 不参与冲突检测
            if (q and q in seen_q) or (q_img and q_img in seen_q_img):
                next_buffer.append(s)
                continue

            # 2. 检查 Pos 冲突 (防止 False Positive)
            # 如果当前样本的任何一个 pos text 已经在 chunk 里出现过 -> 冲突
            if not pos.isdisjoint(seen_p):
                next_buffer.append(s)
                continue
            
            # 如果当前样本的任何一个 pos image 已经在 chunk 里出现过 -> 冲突
            if not pos_img.isdisjoint(seen_p_img):
                next_buffer.append(s)
                continue

            # --- 校验通过，加入 Chunk ---
            
            # 更新样本字段 (set 转回 list)
            s["pos"] = list(pos)
            s["neg"] = list(neg)
            # image 字段保持原样即可，通常是 list 或 str，这里不需修改源数据结构，只需保证逻辑正确

            # 更新状态集合
            if q: seen_q.add(q)
            if q_img: seen_q_img.add(q_img)
            seen_p.update(pos)
            seen_p_img.update(pos_img)
            
            clean_chunk.append(s)

            # chunk 满了就可以输出
            if len(clean_chunk) >= chunk_size:
                break

        yield clean_chunk

        # 更新缓冲区：剩余 (next_buffer) + 循环未遍历到的部分 (buffer[j+1:])
        remaining = buffer[j+1:] if len(clean_chunk) >= chunk_size else [] 
        # 注意：如果上面循环是 break 出来的，remaining 是 buffer[j+1:]
        # 如果是循环正常结束（buffer 耗尽也没满 chunk），remaining 实际上是空，逻辑也自洽
        if len(clean_chunk) < chunk_size:
             # 特殊情况：遍历完了 buffer 还没凑满，剩下的都留在 next_buffer 里等待下一次填充
             # 此时 j 已经到了末尾，remaining 为空
             remaining = []
             
        buffer = next_buffer + remaining


def upsample_to_size(samples, target_size):
    """Upsample samples with replacement until reaching target_size."""
    if not samples:
        return []
    result = list(samples)
    while len(result) < target_size:
        needed = target_size - len(result)
        picks = [random.choice(samples) for _ in range(needed)]
        result.extend(picks)
    return result[:target_size]


def downsample_to_size(samples, target_size):
    """Downsample samples without replacement to target_size."""
    if target_size >= len(samples):
        return list(samples)
    return random.sample(samples, target_size)


def round_to_nearest_multiple(value, multiple):
    """Round value to the nearest multiple of 'multiple'."""
    return max(multiple, int(round(value / multiple)) * multiple)


def apply_sampling_strategy(domain_map, chunk_size, strategy):
    """Apply sampling strategy to each domain."""
    if strategy == "none" or not domain_map:
        return domain_map

    domain_sizes = [len(items) for items in domain_map.values() if items]
    if not domain_sizes:
        return domain_map

    processed = {}

    if strategy == "range":
        q1 = int(np.percentile(domain_sizes, 25))
        q3 = int(np.percentile(domain_sizes, 75))
        q1 = max(q1, chunk_size)
        q3 = max(q3, q1)
        for domain, samples in domain_map.items():
            size = len(samples)
            if size < q1 and size > 0:
                processed[domain] = upsample_to_size(samples, q1)
            elif size > q3:
                processed[domain] = downsample_to_size(samples, q3)
            else:
                processed[domain] = list(samples)

    elif strategy == "median":
        median_size = int(np.median(domain_sizes))
        median_size = max(median_size, chunk_size)
        for domain, samples in domain_map.items():
            size = len(samples)
            if size < median_size and size > 0:
                processed[domain] = upsample_to_size(samples, median_size)
            elif size > median_size:
                processed[domain] = downsample_to_size(samples, median_size)
            else:
                processed[domain] = list(samples)

    elif strategy == "probability":
        for domain, samples in domain_map.items():
            size = len(samples)
            if size == 0:
                processed[domain] = []
                continue
            probability = min(1.0, (size ** 0.75) / size)
            target_size = max(1, int(round(probability * size)))
            processed[domain] = downsample_to_size(samples, target_size)

    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    # round sizes to nearest multiple of chunk_size
    for domain, samples in processed.items():
        target_size = round_to_nearest_multiple(len(samples), chunk_size)
        if len(samples) < target_size:
            processed[domain] = upsample_to_size(samples, target_size)
        elif len(samples) > target_size:
            processed[domain] = downsample_to_size(samples, target_size)

    return processed


def mix_groups_with_chunks_min_variance(all_cls_groups, all_chunks, chunk_size, shuffle=True):
    """
    将分类任务的 group 与其他任务 chunk 混合，使用最小方差装箱策略
    目标：每个输出 chunk 尽量接近 chunk_size
    """
    if shuffle:
        random.shuffle(all_cls_groups)
        random.shuffle(all_chunks)

    # 合并所有 group（分类任务 + 其他任务）
    all_groups = all_cls_groups + all_chunks

    # 按长度从大到小排序，优先放置长的 group
    all_groups.sort(key=len, reverse=True)

    output_bins = []  # 每个 bin 是一个列表，存放若干 group

    for g in all_groups:
        g_len = len(g)
        best_bin = None
        min_increase = float('inf')

        # 尝试放入已有 bin，看哪个最接近目标 chunk_size
        for b in output_bins:
            current_len = sum(len(x) for x in b)
            if current_len + g_len <= chunk_size:
                # 计算放入后的“剩余空间”作为代价
                remaining = chunk_size - (current_len + g_len)
                if remaining < min_increase:
                    min_increase = remaining
                    best_bin = b

        # 若找到合适的 bin，就放进去；否则新建一个 bin
        if best_bin is not None:
            best_bin.append(g)
        else:
            output_bins.append([g])

    # flatten 每个 bin，必要时裁剪到 chunk_size
    output_chunks = []
    for b in output_bins:
        merged = sum(b, [])
        if len(merged) > chunk_size:
            merged = merged[:chunk_size]  # 裁剪到目标长度
        if len(merged) == chunk_size: # 只保留满的 chunk
            output_chunks.append(merged)

    # 打印统计信息
    lengths = [len(c) for c in output_chunks]
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        max_len, min_len = max(lengths), min(lengths)
        variance = sum((x-avg_len)**2 for x in lengths)/len(lengths)
        print(f"[INFO] Generated {len(output_chunks)} chunks.")
        print(f"[INFO] Avg len = {avg_len:.1f}, Max = {max_len}, Min = {min_len}, Var = {variance:.2f}")
    else:
        print("[INFO] No chunks generated.")

    return output_chunks


def main():
    parser = argparse.ArgumentParser(description="Group by domain and shuffle chunks globally.")
    parser.add_argument("input", help="Input .jsonl.gz file")
    parser.add_argument("output", help="Output .jsonl file")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size per domain")
    parser.add_argument(
        "--sampling-strategy",
        choices=["none", "range", "median", "probability"],
        default="none",
        help="Domain-level sampling strategy to balance data sizes",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.parent.exists() is False:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # 从文件名派生seed
    seed = abs(hash(input_path.name)) % (2**32)
    random.seed(seed)
    print(f"[INFO] Using seed: {seed}")

    # 1. 读取数据
    print(f"[INFO] Loading {input_path} ...")
    data = load_jsonl_gz(input_path)
    print(f"[INFO] Loaded {len(data)} samples")

    # 2. 按 source 分组
    domain_map = {}
    for item in data:
        source = item.get("source", "unknown")
        domain_map.setdefault(source, []).append(item)

    # detect domain
    domain_types = detect_domain_type(domain_map, label_threshold=200)

    # apply sampling strategy
    domain_map = apply_sampling_strategy(domain_map, args.chunk_size, args.sampling_strategy)

    # classification domain stratified grouping
    all_cls_domains = []
    for domain, dtype in domain_types.items():
        if dtype == "classification":
            samples = domain_map[domain]
            grouped = stratified_round_robin_grouping(samples)
            if grouped:
                all_cls_domains.append(grouped)
                print(f"[INFO] Domain '{domain}' (stratified): {len(samples)} samples -> {len(grouped)} groups")
            else:
                print(f"[INFO] Domain '{domain}' (stratified): Skipped (empty or no labels)")

    all_cls_groups = []
    for d in all_cls_domains:
        all_cls_groups.extend(d)
    print(f"[INFO] Total classification groups: {len(all_cls_groups)}")

    random.shuffle(all_cls_groups)
    # 3. general domain chunk
    all_chunks = []
    all_remains = []
    for src, items in domain_map.items():
        if domain_types[src] == "classification":
            continue  # 已处理过
        chunks = []
        remaining = []
        for chunk in chunkify(items, args.chunk_size):
            if len(chunk) == args.chunk_size:
                chunks.append(chunk)
            else:
                remaining.append(chunk)
        all_chunks.extend(chunks)
        all_remains.extend(remaining)
        print(f"[INFO] Domain '{src}': {len(chunks)} chunks ({len(items)} samples), {len(remaining)} remaining")
    print(f"[INFO] Total chunks: {len(all_chunks)}, total remaining: {len(all_remains)}")

    # mix classification groups into chunks
    all_chunks_with_remain = all_chunks + all_remains
    output_chunks = mix_groups_with_chunks_min_variance(all_cls_groups, all_chunks_with_remain, args.chunk_size)

    # 4. 全局shuffle chunks
    random.shuffle(output_chunks)
    print(f"[INFO] Final total chunks: {len(output_chunks)}")
    # 5. 输出结果
    print(f"[INFO] Writing output to {output_path}")
    with gzip.open(output_path, "wt", encoding="utf-8") as fout:
        for chunk in output_chunks:
            for item in chunk:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[INFO] Done!")

if __name__ == "__main__":
    main()