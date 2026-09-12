"""Validate complete raw MTEB results and generate the final reproduction report."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/nanobeir_cpu_final'
config=json.loads((OUT/'run_config.json').read_text())
summary=json.loads((OUT/'summary.json').read_text())
assert config['cuda_available'] is False
assert config['backend']=='pure C FP32 CPU'
for filename,key in [('rwkv_emb.c','c_source_sha256'),('rwkv_emb.dll','library_sha256'),
                     ('rwkv_c.py','bridge_sha256'),('evaluate_nanobeir.py','evaluator_sha256'),
                     ('build.ps1','build_sha256')]:
    assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==config[key], filename
with (ROOT/'models/embedding-fp32.bin').open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()==config['weights_sha256']
scores={}
for filename in OUT.rglob('*Retrieval.json'):
    result=json.loads(filename.read_text())
    rows=[r for split in result['scores'].values() for r in split]
    assert len(rows)==1 and math.isfinite(rows[0]['ndcg_at_10'])
    scores[result['task_name']]=rows[0]['ndcg_at_10']*100
assert len(scores)==13 and set(scores)==set(config['tasks'])
assert scores==summary['ndcg_at_10']
mean=sum(scores.values())/13
assert abs(mean-summary['mean_ndcg_at_10'])<1e-10
gpu=json.loads((ROOT.parent/'EmbeddingRWKV/results_nanobeir/baseline/summary.json').read_text())
validation=json.loads((ROOT/'results/numerical_validation.json').read_text())
lines='\n'.join(f'| {task} | {scores[task]:.3f} | {gpu["ndcg_at_10"][task]:.3f} |'
                for task in config['tasks'])
report=f'''# CPU FP32 C 实现与 NanoBEIR 复现

最终完整 13 项 **NDCG@10 × 100 = {mean:.6f}**。

- 与论文目标 59.10 相差 {mean-59.1:+.6f} 分。
- 与此前 GPU BF16 基线 {gpu['mean_ndcg_at_10']:.6f} 相差 {mean-gpu['mean_ndcg_at_10']:+.6f} 分。
- 本轮完整评测耗时 {summary['elapsed_seconds']/60:.2f} 分钟。
- C 实际编码文本数：{summary['c_encoded_texts']}；C 推理耗时 {summary['c_inference_seconds']/60:.2f} 分钟。

## 实现

主工作区：`rwkv-emb.c`。核心推理、模型加载、UTF-8 byte-trie 分词、预处理、RWKV-7 状态更新、
多 EOS 池化和三个任务输出头均在单文件 `rwkv_emb.c` 内，以纯 C 实现。权重、激活、状态和矩阵运算均为 FP32。
未调用 BLAS、PyTorch 或 GPU 推理。OpenMP 及编译器自动向量化用于加速普通 C 循环。
单独运行 `rwkv-emb.exe` 不需要 Python；Windows 可执行文件仅链接 KERNEL32.dll 和 msvcrt.dll。

Python 用于一次性权重导出、对照测试及 MTEB/ctypes 接入。NanoBEIR 进程在导入 PyTorch 前设置
`CUDA_VISIBLE_DEVICES=-1`，并确认 CUDA 不可用。所有检索 embedding 由 C CPU 实现产生。

## 固定评测协议

MTEB {config['mteb']}，全部 13 项默认 split，无样本 limit；batch=4、ctx=2048、EOS chunk=512、
原查询指令、RETR head、原 title/text 拼接规则。包括短文本重复、有效 EOS mask、左侧零填充和 261 对齐填充。
这些预处理与之前 GPU 基线相同；区别在于计算精度和执行后端。

CPU：AMD Ryzen 7 9700X，{config['threads']} OpenMP 线程。编译器：GCC 16.2，
`-Ofast -march=native -mprefer-vector-width=512 -fopenmp`。该构建允许 FP32 重结合及 FMA；没有量化。
权重来自最小的 `rwkv0b1-emb-curriculum.pth`，没有训练或根据 benchmark 标签调整参数。

## 数值与实现验证

- 矩阵乘法方向、非整 tile 尺寸、Unicode 分词、空/长/短文本预处理：通过。
- 16、64、256-token 测试逐层对照原模型的独立 FP32 递推：通过。
- 最终 embedding 最大绝对误差：{max(r['embedding_max_abs'] for r in validation):.9g}。
- CLS/STS/RETR 三头、重复调用状态隔离、shape/NUL 边界检查：通过。
- 独立 C 可执行文件与 ctypes C API 输出比较：完全一致。
- 13 个原始 JSON 的 NDCG@10 重新核算，与汇总一致。
- 已核对本报告的源码、DLL、bridge、评测脚本、构建脚本及实际权重哈希。

## 逐任务结果

| 任务 | C CPU FP32 | GPU BF16 基线 |
|---|---:|---:|
{lines}
| **平均** | **{mean:.6f}** | **{gpu['mean_ndcg_at_10']:.6f}** |

## 重跑及证据

从主工作区执行：

```powershell
.\\build.ps1
..\\EmbeddingRWKV\\.venv\\Scripts\\python.exe test_reference.py
..\\EmbeddingRWKV\\.venv\\Scripts\\python.exe -u evaluate_nanobeir.py --threads 16 --output results/recheck
```

`results/nanobeir_cpu_final/summary.json`：最终汇总；同目录保存逐任务结果、配置及源码快照。
`results/nanobeir_cpu_final.log`：全量运行日志。
`results/test_reference_final.log`、`results/numerical_validation.json`：数值对照。
模型格式、编译和 C API 说明见 `README.md`。重复使用结果目录必须匹配实现/权重/配置身份；改动后使用新目录。

这验证了最小模型的文本 embedding 推理及该 NanoBEIR 协议；不涵盖训练、图像或单独 reranker。
'''
(ROOT/'REPRODUCTION.md').write_text(report,encoding='utf-8')
print(f'Complete CPU result validated: {mean:.9f}; target delta {mean-59.1:+.6f}')
