# rwkv-emb.c

EmbeddingRWKV text embedding inference in a single, readable C file. FP32 weights,
activations, matrix products, recurrent state and output heads run on the CPU.
There is no BLAS, PyTorch, CUDA, quantization, or processor-intrinsic dependency in
the C runtime. Optional OpenMP and compiler auto-vectorization accelerate it.

**Full NanoBEIR result: 58.907231 NDCG@10 x 100**, across all 13 tasks.
This is 0.192769 below the 59.10 target and 0.066462 below the prior GPU BF16
baseline (58.973692). The CPU run encoded 57,372 texts in about 3 h 27 min.
See [REPRODUCTION.md](REPRODUCTION.md) for the per-task comparison and evidence.

The organization follows `karpathy/llm.c/train_gpt2.c`: explicit model structures,
small layer functions, ordinary loops, checked binary loading, and owned buffers.
This implements inference, not training or the separate state reranker/image model.

## Files

- `rwkv_emb.c`: model loading, byte-trie tokenizer, preprocessing, RWKV-7,
  multi-EOS pooling, CLS/STS/RETR heads, normalization, public C API and CLI.
- `export_weights.py`: one-time conversion of the original checkpoint to FP32.
- `rwkv_c.py`: ctypes transport; computation and tokenization remain in C.
- `test_reference.py`: independent FP32 reference comparison using the original
  model equations, including per-layer tensors and final embeddings.
- `evaluate_nanobeir.py`: CPU-only MTEB 1.38.60 adapter. It uses the C runtime for
  every embedding; MTEB performs dataset loading and retrieval metric calculation.
- `build.ps1`: builds the Windows CLI and DLL with GCC, treating warnings as errors.

## Build and run

Windows, from this directory (the local compiler is in `.tools/w64devkit`):

```powershell
.\build.ps1
.\rwkv-emb.exe models/embedding-fp32.bin models/tokenizer.bin "What is machine learning?"
```

The executable writes one normalized 768-dimensional JSON vector to stdout.
Use relative ASCII model paths on Windows; the current directory may contain
non-ASCII characters. The API takes UTF-8 text. The ctypes API supports multilingual
text; Windows command-line argument encoding depends on the invoking shell.

Portable GCC builds (OpenMP optional; no machine-specific intrinsics in source):

```sh
gcc -O3 -fopenmp rwkv_emb.c -lm -o rwkv-emb
gcc -O3 -fopenmp -shared -fPIC -DRWKV_NO_MAIN rwkv_emb.c -lm -o librwkv_emb.so
```

The evaluated Windows build uses GCC 16.2, `-Ofast -march=native
-mprefer-vector-width=512 -fopenmp`. This allows FP32 reassociation and FMA, as in
llm.c's optimized reference build. It does not change values to FP16/BF16.
The `-march=native` binaries should be rebuilt on a different CPU.

## Weights and binary format

The included local export originates from
`../EmbeddingRWKV/models/rwkv0b1-emb-curriculum.pth`.
Re-export using the existing Python environment:

```powershell
..\EmbeddingRWKV\.venv\Scripts\python.exe export_weights.py
```

`models/embedding-fp32.bin` contains 410 text-model tensors (~577 MB). It excludes
the image model and unused language-model vocabulary output projection, and
includes all three embedding heads. `models/manifest.json` records source/export
SHA256 digests and shapes. There is no retraining or fitting to benchmark labels.

The format is little-endian: eight uint32 values (magic `0x52454D42`, version 1,
layers, channels, vocabulary size, head size, tensor count, reserved), followed by
each tensor's 128-byte zero-terminated ASCII name, uint32 rank, four uint32 dimensions,
uint64 element count, and contiguous float32 data. Matrices use `[input,output]`;
the embedding table uses `[token,channel]`. The loader makes a lossless one-time
copy into 32-column matrix tiles for better cache access.

The tokenizer file contains magic `0x52564F43`, uint32 entry count, then repeated
uint32 token ID, uint32 byte length, and raw bytes. Vocabulary strings are exported
with `ast.literal_eval`. C uses longest-prefix byte matching. Embedded NUL is not
supported by the C string API and is rejected by the Python bridge.

## Numerical verification

```powershell
..\EmbeddingRWKV\.venv\Scripts\python.exe test_reference.py
```

Tests cover matrix orientation and non-tile dimensions; ASCII/Chinese/Unicode
tokenization; empty, long, repeated and padded text preprocessing; all 12 RWKV
blocks; output normalization; three task heads; and independent repeated calls.
The FP32 reference is the original model with a separate PyTorch FP32 recurrence.
GPU use is limited to this reference test, never C inference or NanoBEIR inference.
See `results/numerical_validation.json` and `results/test_reference_final.log`.

## NanoBEIR reproduction

```powershell
..\EmbeddingRWKV\.venv\Scripts\python.exe -u evaluate_nanobeir.py --threads 16 --output results/nanobeir_cpu_final
```

The fixed protocol matches the prior 58.9737 GPU baseline: all 13 NanoBEIR tasks,
batch size 4, context 2048, EOS chunk 512, original generic query instruction,
RETR head, and the original title/text joining. No samples are truncated with an
evaluation `limit`. The model's documented token context cap is preserved.
Short text repeats, EOS selection, left zeros and 261 alignment tokens intentionally
match the original evaluation wrapper, including its batching semantics.

The adapter sets `CUDA_VISIBLE_DEVICES=-1` before importing PyTorch and verifies
CUDA is unavailable. Python is only the benchmark/FFI harness; all embeddings are
computed by the C FP32 model on the CPU. Dataset and dependency caches reuse the
previous reproduction environment.

Each run saves configuration/source/library/weight identities, individual MTEB
result JSONs, progress and the final mean NDCG@10 x 100. Reusing an output directory
is permitted only if all recorded identities match. Use a new output directory
after implementation or configuration changes. An interrupted unfinished task is
recomputed; completed tasks can be resumed without mixing different implementations.

The full CPU run completed successfully. The measured result is recorded in
`results/nanobeir_cpu_final/summary.json` and `REPRODUCTION.md`. The 13 raw task
scores, their mean, and actual implementation/weight hashes were independently
rechecked after completion. No evaluation configuration was tuned to fit the target.

## Scope and memory

Text embeddings only; all three task heads are exposed (`0=CLS`, `1=STS`, `2=RETR`).
The standalone demo selects RETR. The recurrent state is reset for each sequence.
Temporary layer buffers scale with batch token count; the recurrent state itself
is one 64x64 FP32 matrix per active head. Weights occupy ~577 MB plus small metadata.
The loader/API handle valid exported models; fatal corrupt-file/allocation errors
report a message and terminate, following the reference-program style of llm.c.
Null text pointers and malformed tokenizer/preprocessing input also terminate;
the Python bridge rejects shape mismatches and embedded NUL before calling C.

The model and tokenizer originate from howard-hou/EmbeddingRWKV (Apache-2.0).
The original repository is retained alongside this workspace. No llm.c source
has been copied; it is an organization/style reference.
