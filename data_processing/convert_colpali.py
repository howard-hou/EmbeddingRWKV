import os
import json
from datasets import load_from_disk
from pathlib import Path

# 1. 设置根目录
output_root_dir = "Text2VD/colpali"      # 所有图片的根目录
output_jsonl_file = "colpali.jsonl" 
os.makedirs(output_root_dir, exist_ok=True)

# 假设 ds 已经加载
ds = load_from_disk("colpali_train_set")

# =======================================================
# 步骤 1: 添加全局索引 (为了生成唯一的 ID)
# =======================================================
ds = ds.add_column("row_idx", range(len(ds)))

# =======================================================
# 步骤 2: 定义核心处理函数
# =======================================================
def process_entry(example):
    # --- 准备工作 ---
    source_val = str(example['source']) # 拿到 source，例如 "arxiv" 或 "finance"
    image = example['image']
    raw_filename = Path(example['image_filename'])
    idx = example['row_idx']
    
    # --- A. 路径处理 (关键修改) ---
    # 1. 提取文件名并去后缀
    name_no_ext = raw_filename.stem
    if source_val == "pdf":
        pp = raw_filename.parent.parent.name  # 获取上上级目录名作为 source_val
        p = raw_filename.parent.stem      # 获取上级目录名作为文件名的一部分
        sub_dir = Path(output_root_dir) / f"{pp}_{source_val}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        full_save_path = sub_dir / f"{p}_{name_no_ext}.jpg"
        full_save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        sub_dir = Path(output_root_dir) / source_val
        sub_dir.mkdir(parents=True, exist_ok=True)
        new_filename = f"{name_no_ext}.jpg"
        full_save_path = sub_dir / new_filename
    
    # --- B. 保存图片 ---
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(full_save_path, format='JPEG', quality=95)
    
    # --- C. 构建 JSONL 字段 ---
    
    # 构造 ID
    unique_id = f"t2vd_colpali_{source_val}_{idx}"
    
    # 构造 pos_image 路径 (相对路径或完整路径，这里保留目录结构)
    # 结果类似于: "colpali/science/doc_01.jpg"
    rel_image_path = str(full_save_path)
    
    return {
        "id": unique_id,
        "instruct": "Find a screenshot that relevant to the user’s question.",
        "query": example['query'],
        "query_image": "",
        "pos": [],
        "pos_image": [rel_image_path],  # 记录带有 source 子目录的路径
        "neg": [],
        "neg_image": [],
        "source": f"colpali_{source_val}",
        "task": "[RETR]"
    }

# =======================================================
# 步骤 3: 多进程并行处理
# =======================================================
# remove_columns 确保内存不爆炸
new_ds = ds.map(
    process_entry,
    num_proc=16, 
    remove_columns=ds.column_names, 
    desc="Processing images by source"
)

# =======================================================
# 步骤 4: 保存结果
# =======================================================
print(f"处理完成，JSONL 正在写入 {output_jsonl_file} ...")
with open(output_jsonl_file, 'w', encoding='utf-8') as f_out:
    for record in new_ds:
        f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
print("Done!")