"""One-time export. The C inference runtime has no PyTorch dependency."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import torch

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=ROOT.parent / 'EmbeddingRWKV/models/rwkv0b1-emb-curriculum.pth')
    args = p.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    tensors = {}
    for key, value in state.items():
        if not (key.startswith('rwkv.') or key.startswith('head.')):
            continue
        if key == 'rwkv.head.weight':
            continue
        value = value.float()
        # All matrix products in C use row-major W[input, output].
        # nn.Linear stores [output,input]; LoRA matrices already use [input,output].
        if value.ndim == 2 and key.endswith('.weight') and key != 'rwkv.emb.weight':
            value = value.T
        tensors[key] = value.contiguous().numpy()
    C = state['rwkv.emb.weight'].shape[1]
    L = max(int(k.split('.')[2]) for k in state if k.startswith('rwkv.blocks.')) + 1
    dest = ROOT / 'models/embedding-fp32.bin'
    dest.parent.mkdir(exist_ok=True)
    with dest.open('wb') as f:
        f.write(struct.pack('<8I', 0x52454D42, 1, L, C, 65536, 64, len(tensors), 0))
        for key, value in tensors.items():
            name = key.encode('ascii')
            assert len(name) < 128
            f.write(name.ljust(128, b'\0'))
            f.write(struct.pack('<5IQ', value.ndim, *(list(value.shape)+[1]*4)[:4], value.size))
            f.write(value.astype('<f4', copy=False).tobytes())
    vocab_path = ROOT.parent / 'EmbeddingRWKV/embedding/eval/tokenizer/rwkv_vocab_v20230424.txt'
    vocab = []
    for line in vocab_path.read_text(encoding='utf-8').splitlines():
        start, end = line.index(' '), line.rindex(' ')
        value = ast.literal_eval(line[start:end])
        if isinstance(value, str):
            value = value.encode('utf-8')
        assert len(value) == int(line[end:])
        vocab.append((int(line[:start]), value))
    with (ROOT / 'models/tokenizer.bin').open('wb') as f:
        f.write(struct.pack('<2I', 0x52564F43, len(vocab)))
        for index, value in vocab:
            f.write(struct.pack('<2I', index, len(value)))
            f.write(value)
    with args.checkpoint.open('rb') as f:
        source_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    with dest.open('rb') as f:
        target_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    meta = dict(source=str(args.checkpoint), source_sha256=source_hash, sha256=target_hash,
                layers=L, channels=C, dtype='float32', matrix_layout='input,output',
                tensors={key:list(value.shape) for key,value in tensors.items()})
    (ROOT / 'models/manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Exported {len(tensors)} tensors: {dest.stat().st_size} bytes')

if __name__ == '__main__':
    main()
