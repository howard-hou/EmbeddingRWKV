"""MTEB drives the benchmark; CPU C performs every model operation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HOME']=str(ROOT.parent/'EmbeddingRWKV/.cache/huggingface')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
import numpy as np
import torch
import mteb
from mteb.encoder_interface import PromptType
from rwkv_c import CModel

INSTRUCTION='Instruct: Given a query, retrieve documents that answer the query\nQuery: {query}'

class MTEBCModel:
    def __init__(self,threads,output):
        self.c=CModel(threads); self.output=output; self.total_texts=0; self.total_seconds=0
    def encode(self,sentences,task_name,prompt_type=None,batch_size=4,**kwargs):
        if prompt_type==PromptType.query:
            if sentences and 'Instruct:' not in sentences[0]:
                sentences=[INSTRUCTION.format(query=s) for s in sentences]
        outputs=[]; start=time.time(); count=len(sentences)
        for i in range(0,count,batch_size):
            texts=sentences[i:i+batch_size]
            vec=self.c.encode(texts,ctx=2048,chunk=512,task=2)
            if not np.isfinite(vec).all(): raise RuntimeError('Nonfinite C embeddings')
            outputs.append(vec)
            if i==0 or (i//batch_size+1)%16==0 or i+batch_size>=count:
                done=min(i+batch_size,count); elapsed=time.time()-start
                status=dict(task=task_name,done=done,total=count,elapsed_seconds=elapsed,
                            estimated_remaining_seconds=elapsed*(count-done)/done,
                            texts_per_second=done/elapsed)
                (self.output/'progress.json').write_text(json.dumps(status,indent=2))
                print(f'{task_name}: {done}/{count} texts, {elapsed:.1f}s, ETA {status["estimated_remaining_seconds"]:.0f}s',flush=True)
        self.total_texts+=count; self.total_seconds+=time.time()-start
        return np.concatenate(outputs) if outputs else np.empty((0,768),np.float32)
    def encode_queries(self,queries,task_name,batch_size=4,**kwargs):
        queries=[INSTRUCTION.format(query=q) for q in queries]
        return self.encode(queries,task_name,batch_size=batch_size,**kwargs)
    def encode_corpus(self,corpus,batch_size=4,**kwargs):
        texts=[((doc.get('title') or '')+' '+(doc.get('text') or '')).strip()
               if isinstance(doc,dict) else doc for doc in corpus]
        return self.encode(texts,batch_size=batch_size,**kwargs)
    def similarity(self,a,b):
        a=torch.as_tensor(a,dtype=torch.float32,device='cpu')
        b=torch.as_tensor(b,dtype=torch.float32,device='cpu')
        return a.reshape(-1,768)@b.reshape(-1,768).T

def main():
    p=argparse.ArgumentParser(); p.add_argument('--threads',type=int,default=16)
    p.add_argument('--task'); p.add_argument('--output',default='results/nanobeir_fp32')
    args=p.parse_args(); out=ROOT/args.output; out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    assert not torch.cuda.is_available(), 'Evaluation must be CPU-only'
    with (ROOT/'models/embedding-fp32.bin').open('rb') as f:
        weights_hash=hashlib.file_digest(f,'sha256').hexdigest()
    if weights_hash!=json.loads((ROOT/'models/manifest.json').read_text())['sha256']:
        raise RuntimeError('Actual weights do not match export manifest')
    model=MTEBCModel(args.threads,out)
    tasks=mteb.get_tasks(tasks=[args.task]) if args.task else mteb.get_benchmark('NanoBEIR').tasks
    config=dict(backend='pure C FP32 CPU',threads=args.threads,batch_size=4,ctx_len=2048,eos_chunk_size=512,
                instruction=INSTRUCTION,mteb=mteb.__version__,tasks=[t.metadata.name for t in tasks],
                c_source_sha256=hashlib.sha256((ROOT/'rwkv_emb.c').read_bytes()).hexdigest(),
                library_sha256=hashlib.sha256((ROOT/'rwkv_emb.dll').read_bytes()).hexdigest(),
                bridge_sha256=hashlib.sha256((ROOT/'rwkv_c.py').read_bytes()).hexdigest(),
                evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                build_sha256=hashlib.sha256((ROOT/'build.ps1').read_bytes()).hexdigest(),
                weights_sha256=weights_hash,
                numpy=np.__version__,torch=torch.__version__,python=sys.version,
                tokenizer_sha256=hashlib.sha256((ROOT/'models/tokenizer.bin').read_bytes()).hexdigest(),
                cuda_available=torch.cuda.is_available())
    config_path=out/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise RuntimeError('Run identity changed; use a new --output directory to avoid stale scores')
    if not config_path.exists() and list(out.rglob('*Retrieval.json')):
        raise RuntimeError('Existing scores have no run identity; use a new --output directory')
    config_path.write_text(json.dumps(config,indent=2))
    start=time.time()
    results=mteb.MTEB(tasks=tasks).run(model,encode_kwargs={'batch_size':4},output_folder=str(out),overwrite_results=False)
    scores={r.task_name:float(r.get_score())*100 for r in results}
    summary=dict(ndcg_at_10=scores,mean_ndcg_at_10=sum(scores.values())/len(scores),
                 elapsed_seconds=time.time()-start,c_encoded_texts=model.total_texts,c_inference_seconds=model.total_seconds)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True); model.c.close()

if __name__=='__main__': main()
