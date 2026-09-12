import argparse
import time
import numpy as np
from rwkv_c import CModel,F,lib
p=argparse.ArgumentParser(); p.add_argument('--threads',type=int,default=8); p.add_argument('--tokens',type=int,default=128)
a=p.parse_args(); c=CModel(a.threads)
for M,K,N in [(512,768,768),(512,768,3072),(512,3072,768)]:
    x=np.ones((M,K),np.float32); w=np.ones((K,N),np.float32); y=np.empty((M,N),np.float32)
    lib.rwkv_matmul(y.ctypes.data_as(F),x.ctypes.data_as(F),w.ctypes.data_as(F),M,K,N)
    start=time.perf_counter()
    for _ in range(3): lib.rwkv_matmul(y.ctypes.data_as(F),x.ctypes.data_as(F),w.ctypes.data_as(F),M,K,N)
    seconds=(time.perf_counter()-start)/3
    print('matmul',M,K,N,round(seconds,4),'seconds',round(2*M*K*N/seconds/1e9,2),'GFLOPS',flush=True)
ids=np.full((4,a.tokens),187,np.int32); ids[:,-1]=65535; mask=(ids==65535).astype(np.uint8)
start=time.perf_counter(); out=c.tokens(ids,mask); seconds=time.perf_counter()-start
print('C forward',ids.shape,round(seconds,3),'seconds',round(ids.size/seconds,2),'tokens/s',flush=True)
c.close()
