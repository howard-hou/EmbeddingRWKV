import os
import sys
import torch

def merge_ckpts(folder):
    ckpt_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".pth") or f.endswith(".pt")
    ]
    assert ckpt_files, "没有找到任何 ckpt 文件"

    print(f"找到 {len(ckpt_files)} 个 checkpoint")

    avg_state = None
    for i, ckpt_file in enumerate(ckpt_files):
        state = torch.load(ckpt_file, map_location="cpu")
        if avg_state is None:
            # 保持原始精度
            avg_state = {k: v.clone() for k, v in state.items()}
        else:
            for k in avg_state.keys():
                avg_state[k] += state[k].to(avg_state[k].dtype)
        print(f"已处理: {ckpt_file}")

    # 平均化
    n = len(ckpt_files)
    for k in avg_state.keys():
        avg_state[k] /= n

    out_path = os.path.join(folder, "merged.pth")
    torch.save(avg_state, out_path)
    print(f"已保存到: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python merge_ckpt.py <ckpt文件夹路径>")
        sys.exit(1)
    merge_ckpts(sys.argv[1])
