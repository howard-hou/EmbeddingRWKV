"""ctypes transport only: all tokenization, preprocessing and inference run in C."""
import ctypes as ct
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
if os.name == 'nt':
    _dll_dir = os.add_dll_directory(str(ROOT / '.tools/w64devkit/bin'))
lib = ct.CDLL(str(ROOT / ('rwkv_emb.dll' if os.name=='nt' else 'librwkv_emb.so')))
F = ct.POINTER(ct.c_float)
I = ct.POINTER(ct.c_int32)
U = ct.POINTER(ct.c_uint8)
P = ct.c_void_p
lib.rwkv_threads.argtypes=[ct.c_int]
lib.rwkv_load.argtypes=[ct.c_char_p]; lib.rwkv_load.restype=P
lib.rwkv_free.argtypes=[P]
lib.rwkv_channels.argtypes=[P]; lib.rwkv_channels.restype=ct.c_int
lib.rwkv_layers.argtypes=[P]; lib.rwkv_layers.restype=ct.c_int
lib.rwkv_load_tokenizer.argtypes=[P,ct.c_char_p]
lib.rwkv_tokenize.argtypes=[P,ct.c_char_p,I,ct.c_int]
lib.rwkv_prepare.argtypes=[P,ct.POINTER(ct.c_char_p),ct.c_int,ct.c_int,ct.c_int,ct.POINTER(I),ct.POINTER(U)]
lib.rwkv_release.argtypes=[P]
lib.rwkv_encode_tokens.argtypes=[P,I,U,ct.c_int,ct.c_int,ct.c_int,F,F]
lib.rwkv_encode_texts.argtypes=[P,ct.POINTER(ct.c_char_p),ct.c_int,ct.c_int,ct.c_int,ct.c_int,F]
lib.rwkv_matmul.argtypes=[F,F,F,ct.c_int,ct.c_int,ct.c_int]

class CModel:
    def __init__(self, threads=8):
        lib.rwkv_threads(threads)
        # C fopen uses the platform's narrow encoding on Windows. Relative ASCII
        # model paths work even if the working directory contains non-ASCII names.
        previous=os.getcwd()
        try:
            os.chdir(ROOT)
            self.handle=lib.rwkv_load(b'models/embedding-fp32.bin')
            if not self.handle: raise RuntimeError('Cannot load C model')
            if lib.rwkv_load_tokenizer(self.handle,b'models/tokenizer.bin'):
                raise RuntimeError('Cannot load tokenizer')
        finally:
            os.chdir(previous)
        self.channels=lib.rwkv_channels(self.handle)
        self.layers=lib.rwkv_layers(self.handle)
    def close(self):
        if self.handle: lib.rwkv_free(self.handle); self.handle=None
    def tokenize(self,text):
        if '\0' in text: raise ValueError('Embedded NUL is not supported by the C string API')
        data=text.encode('utf-8'); out=np.empty(len(data)+1,np.int32)
        n=lib.rwkv_tokenize(self.handle,data,out.ctypes.data_as(I),len(out))
        if n<0: raise RuntimeError('Tokenizer failed')
        return out[:n]
    def prepare(self,texts,ctx=2048,chunk=512):
        if not texts or any('\0' in t for t in texts): raise ValueError('Empty batch or embedded NUL')
        array=(ct.c_char_p*len(texts))(*(t.encode('utf-8') for t in texts))
        ids,mask=I(),U()
        T=lib.rwkv_prepare(self.handle,array,len(texts),ctx,chunk,ct.byref(ids),ct.byref(mask))
        if T<0: raise RuntimeError('Preprocessing failed')
        try:
            return (np.ctypeslib.as_array(ids,(len(texts)*T,)).copy().reshape(len(texts),T),
                    np.ctypeslib.as_array(mask,(len(texts)*T,)).copy().reshape(len(texts),T))
        finally: lib.rwkv_release(ids); lib.rwkv_release(mask)
    def tokens(self,ids,mask,task=2,trace=False):
        ids=np.ascontiguousarray(ids,dtype=np.int32); mask=np.ascontiguousarray(mask,dtype=np.uint8)
        if ids.shape!=mask.shape or ids.ndim!=2:
            raise ValueError('ids and mask must have equal [B,T] shapes')
        B,T=ids.shape; out=np.empty((B,self.channels),np.float32)
        traces=np.empty((self.layers+2,B,T,self.channels),np.float32) if trace else None
        status=lib.rwkv_encode_tokens(self.handle,ids.ctypes.data_as(I),mask.ctypes.data_as(U),B,T,task,
                                      out.ctypes.data_as(F),traces.ctypes.data_as(F) if trace else None)
        if status: raise RuntimeError(f'C inference failed: {status}')
        return (out,traces) if trace else out
    def encode(self,texts,ctx=2048,chunk=512,task=2):
        if not texts: return np.empty((0,self.channels),np.float32)
        if any('\0' in t for t in texts): raise ValueError('Embedded NUL is not supported by the C string API')
        array=(ct.c_char_p*len(texts))(*(t.encode('utf-8') for t in texts))
        out=np.empty((len(texts),self.channels),np.float32)
        status=lib.rwkv_encode_texts(self.handle,array,len(texts),ctx,chunk,task,out.ctypes.data_as(F))
        if status: raise RuntimeError(f'C text inference failed: {status}')
        return out
