# 评测使用说明

## 环境依赖

- Python 3.10+
- 安装 MTEB：`pip install mteb==1.38.60`
- 安装 tabulate（用于表格显示）：`pip install "tabulate>=0.9.0"`

## MTEB 评测

使用 `mteb_runner.py` 对模型在 MTEB 基准上进行评测，示例命令如下：

```bash
python mteb_runner.py \
  --model-path /path/to/ckpt.pth \
  --vision-tower-path /path/to/vision_tower \
  --benchmark_name MTEB_ENG_V2 \
  --batch-size 8 \
  --ctx-len 1024 \
  --n-layer 12 \
  --n-embd 768
```

### Bash 脚本

仓库提供了一个脚本可简化以上流程：

```bash
bash scripts/run_mteb.sh /path/to/ckpt.pth MTEB_ENG_V2 cuda:1
```

这些脚本展示了在不同数据集上评测模型的配置方式，可根据需要进行修改。
