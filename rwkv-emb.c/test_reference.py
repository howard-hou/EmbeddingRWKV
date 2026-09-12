"""Compare pure C to the original model's FP32 equations, layer by layer."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from rwkv_c import CModel, ROOT, F, lib

REF=ROOT.parent / 'EmbeddingRWKV/embedding/eval'
sys.path.insert(0,str(REF))
os.environ['RWKV_JIT_ON']='0'; os.environ['RWKV_HEAD_SIZE_A']='64'
os.environ['CUPY_CACHE_DIR']=str(ROOT.parent/'EmbeddingRWKV/.cache/cupy')
from nanobeir_nvrtc import import_original_wrapper

def fp32_recurrence(q,w,k,v,a,b):
    B,T,C=q.shape; H=C//64
    q,w,k,v,a,b=[x.reshape(B,T,H,64) for x in (q,w,k,v,a,b)]
    state=torch.zeros(B,H,64,64,device=q.device,dtype=torch.float32)
    outputs=[]
    for t in range(T):
        s_a=(state*a[:,t].unsqueeze(-2)).sum(-1)
        state=(state*torch.exp(-torch.exp(w[:,t])).unsqueeze(-2)
               +s_a.unsqueeze(-1)*b[:,t].unsqueeze(-2)
               +v[:,t].unsqueeze(-1)*k[:,t].unsqueeze(-2))
        outputs.append((state*q[:,t].unsqueeze(-2)).sum(-1))
    return torch.stack(outputs,1).reshape(B,T,C)

def main():
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    c=CModel(8)
    rng=np.random.default_rng(42)
    # Matrix orientation, non-tile dimensions, and FP32 accumulation.
    for M,K,N in [(3,7,5),(8,64,64),(17,768,128)]:
        x=rng.standard_normal((M,K),dtype=np.float32); w=rng.standard_normal((K,N),dtype=np.float32)
        y=np.empty((M,N),np.float32)
        lib.rwkv_matmul(y.ctypes.data_as(F),x.ctypes.data_as(F),w.ctypes.data_as(F),M,K,N)
        np.testing.assert_allclose(y,x@w,atol=0.0001,rtol=0.0001)
    print('Matmul tests PASS',flush=True)
    wrapper=import_original_wrapper()
    cfg=wrapper.VisualRWKVMTEBConfig(
        model_path=str(ROOT.parent/'EmbeddingRWKV/models/rwkv0b1-emb-curriculum.pth'),
        vision_tower_path=str(ROOT.parent/'EmbeddingRWKV/models/siglip2-base-patch16-256'),
        n_layer=12,n_embd=768,ctx_len=2048,eos_chunk_size=512)
    os.chdir(REF)
    original=wrapper.VisualRWKVMTEBModel(cfg)
    original.model.float()
    import src.model as equations
    equations.RUN_CUDA_RWKV7g=fp32_recurrence
    texts=['','Hello world!','你好，世界！','naïve café 🧠\nline two',
           'abc '*510,'very long passage. '*3000]
    for text in texts:
        assert c.tokenize(text).tolist()==original.tokenizer.encode(text)
    for batch in [texts[:4],texts[4:],['x','word '*600,'a '*17000]]:
        ids,mask=c.prepare(batch)
        original_batch=original._build_batch(batch)
        ids_ref=original_batch['query_ids'].cpu().numpy()
        emask=original_batch['query_eos_mask'].cpu().numpy()
        mask_ref=np.zeros_like(ids_ref,np.uint8)
        for row in range(len(batch)): mask_ref[row,ids_ref[row]==65535]=emask[row]
        padding=(-ids_ref.shape[1])%16
        ids_ref=np.pad(ids_ref,((0,0),(padding,0)),constant_values=261)
        mask_ref=np.pad(mask_ref,((0,0),(padding,0)))
        np.testing.assert_array_equal(ids,ids_ref)
        np.testing.assert_array_equal(mask,mask_ref)
    print('UTF-8 tokenizer and short/long/padded preprocessing PASS',flush=True)
    results=[]
    for T in [16,64,256]:
        ids=rng.integers(1,60000,(2,T),dtype=np.int32)
        ids[:,T//2-1]=65535; ids[:,-1]=65535
        mask=(ids==65535).astype(np.uint8)
        before=time.perf_counter(); got,traces=c.tokens(ids,mask,trace=True)
        duration=time.perf_counter()-before
        tid=torch.tensor(ids.astype(np.int64),device='cuda'); tm=torch.tensor(mask.astype(bool),device='cuda')
        collected=[]; hooks=[]
        for block in original.model.rwkv.blocks:
            hooks.append(block.register_forward_hook(lambda mod,inp,out: collected.append(out[0].detach().cpu().numpy())))
        with torch.inference_mode():
            embeddings=original.model.rwkv.emb(tid)
            final=original.model.rwkv(embeddings)
            pool=(final*tm.unsqueeze(-1)).sum(1)/tm.sum(1,keepdim=True)
            ref=original.model.head.retr_head(pool)
            ref=torch.nn.functional.normalize(ref,dim=-1).cpu().numpy()
        for hook in hooks: hook.remove()
        refs=[embeddings.cpu().numpy(),*collected,final.cpu().numpy()]
        errors=[]
        for layer,(actual,expected) in enumerate(zip(traces,refs)):
            error=float(np.max(np.abs(actual-expected))); errors.append(error)
            print(f'T={T} trace {layer:2}: max abs {error:.7g}',flush=True)
            np.testing.assert_allclose(actual,expected,atol=0.003,rtol=0.003)
        np.testing.assert_allclose(got,ref,atol=0.0001,rtol=0.001)
        cosine=np.sum(got*ref,axis=1)
        assert np.min(cosine)>0.99999
        results.append(dict(T=T,c_seconds=duration,trace_max_abs=errors,
                            embedding_max_abs=float(np.max(np.abs(got-ref))),cosine=cosine.tolist()))
        print('Embedding PASS',results[-1],flush=True)
    # Different heads and repeated calls must not retain recurrent state.
    np.testing.assert_array_equal(c.tokens(ids,mask),got)
    # Explicit API boundary checks remain active under python -O.
    try:
        c.tokens(ids,mask[:,:-1])
        raise AssertionError('Mismatched mask was accepted')
    except ValueError:
        pass
    try:
        c.encode(['a\0b'])
        raise AssertionError('Embedded NUL was accepted')
    except ValueError:
        pass
    for task,head in [(0,original.model.head.cls_head),(1,original.model.head.sts_head)]:
        with torch.inference_mode(): ref=torch.nn.functional.normalize(head(pool),dim=-1).cpu().numpy()
        np.testing.assert_allclose(c.tokens(ids,mask,task),ref,atol=0.0001,rtol=0.001)
    (ROOT/'results/numerical_validation.json').write_text(json.dumps(results,indent=2))
    c.close(); print('ALL REFERENCE TESTS PASS',flush=True)

if __name__=='__main__': main()
