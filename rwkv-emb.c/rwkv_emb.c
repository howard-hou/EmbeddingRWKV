/*
EmbeddingRWKV inference in plain C, FP32, CPU only.
Style reference: karpathy/llm.c/train_gpt2.c (small functions, explicit buffers).
No BLAS, framework, processor intrinsics, or GPU dependency. OpenMP is optional.
Weights are row-major float32. Matrix weights are stored [input, output].

Build: gcc -O3 -march=native -fopenmp -ffp-contract=fast rwkv_emb.c -lm -o rwkv-emb
DLL:   gcc <same flags> -shared -DRWKV_NO_MAIN rwkv_emb.c -lm -o rwkv_emb.dll
Run:   rwkv-emb models/embedding-fp32.bin models/tokenizer.bin "Your UTF-8 text"
*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_WIN32) && defined(RWKV_NO_MAIN)
#define API __declspec(dllexport)
#else
#define API
#endif

#define EOS 65535
#define ALIGN_TOKEN 261
#define MAX_LAYERS 64
#define MAX_TENSORS 4096

static void *allocate(size_t count, size_t size) {
    if (!count || size > SIZE_MAX / count) { fprintf(stderr,"Invalid allocation\n"); exit(1); }
    void *p = calloc(count, size);
    if (!p) { fprintf(stderr,"Out of memory (%zu bytes)\n", count*size); exit(1); }
    return p;
}
static void read_exact(void *p, size_t size, size_t count, FILE *f) {
    if (fread(p, size, count, f) != count) { fprintf(stderr,"Truncated model/tokenizer\n"); exit(1); }
}
static void fail(const char *message) { fprintf(stderr,"%s\n",message); exit(1); }

typedef struct { char name[128]; uint32_t ndim, dims[4]; uint64_t count; float *data; int packed; } Tensor;
typedef struct {
    float *ln1w,*ln1b,*ln2w,*ln2b;
    float *mix[6], *w0,*w1,*w2,*a0,*a1,*a2,*v0,*v1,*v2,*g1,*g2;
    float *kk,*ka,*rk,*r,*k,*v,*o,*gnw,*gnb,*fmix,*fk,*fv;
    int dw, da, dv, dg, ff;
} Layer;
typedef struct { float *fc1,*fc2,*nw,*nb; } Head;
typedef struct { int child, sibling, token; unsigned char byte; } TrieNode;
typedef struct {
    int layers, channels, vocab, head_size, ntensors;
    Tensor *tensors;
    float *embedding,*ln0w,*ln0b,*lnw,*lnb;
    Layer block[MAX_LAYERS]; Head heads[3];
    TrieNode *trie; int ntrie, trie_capacity;
} Model;

static Tensor *tensor(Model *m, const char *name) {
    for (int i=0;i<m->ntensors;i++) if (!strcmp(m->tensors[i].name,name)) return m->tensors+i;
    fprintf(stderr,"Missing tensor: %s\n",name); exit(1);
}
/* Packing is a one-time copy, not quantization. Consecutive K rows in a 32-column
   tile are contiguous, avoiding page-strided reads of large linear matrices. */
static float *pack_matrix(const float *w,int K,int N) {
    int tiles=(N+31)/32;
    float *packed=allocate((size_t)tiles*K*32,sizeof(float));
    #pragma omp parallel for
    for(int tile=0;tile<tiles;tile++) for(int k=0;k<K;k++) {
        int count=N-tile*32; if(count>32) count=32;
        memcpy(packed+((size_t)tile*K+k)*32,w+(size_t)k*N+tile*32,count*sizeof(float));
    }
    return packed;
}
static void pack_tensor(Tensor *t) {
    if(t->packed || t->ndim!=2) return;
    float *p=pack_matrix(t->data,t->dims[0],t->dims[1]); free(t->data); t->data=p; t->packed=1;
}
static float *parameter(Model *m, const char *name, size_t count) {
    Tensor *t=tensor(m,name);
    if (t->count != count) { fprintf(stderr,"Wrong shape: %s\n",name); exit(1); }
    if(t->ndim==2 && strcmp(name,"rwkv.emb.weight")) pack_tensor(t);
    return t->data;
}
static Tensor *layer_tensor(Model *m, int i, const char *suffix) {
    char name[128]; snprintf(name,sizeof(name),"rwkv.blocks.%d.%s",i,suffix);
    return tensor(m,name);
}
static float *layer_param(Model *m, int i, const char *suffix, size_t n) {
    Tensor *t=layer_tensor(m,i,suffix);
    if (t->count!=n) { fprintf(stderr,"Wrong layer shape: %s\n",t->name); exit(1); }
    if(t->ndim==2 && strcmp(suffix,"att.r_k")) pack_tensor(t);
    return t->data;
}

API void rwkv_threads(int n) {
#ifdef _OPENMP
    if (n>0) { omp_set_dynamic(0); omp_set_num_threads(n); }
#else
    (void)n;
#endif
}

API Model *rwkv_load(const char *path) {
    FILE *f=fopen(path,"rb"); if (!f) { perror(path); return NULL; }
    uint32_t h[8]; read_exact(h,4,8,f);
    if (h[0]!=0x52454d42 || h[1]!=1 || !h[2] || h[2]>MAX_LAYERS ||
        !h[3] || h[3]>8192 || h[4]!=65536 || h[5]!=64 || h[3]%h[5] ||
        !h[6] || h[6]>MAX_TENSORS) fail("Invalid checkpoint header");
    Model *m=allocate(1,sizeof(*m));
    m->layers=h[2]; m->channels=h[3]; m->vocab=h[4]; m->head_size=h[5]; m->ntensors=h[6];
    m->tensors=allocate(m->ntensors,sizeof(Tensor));
    for (int i=0;i<m->ntensors;i++) {
        Tensor *t=m->tensors+i; read_exact(t->name,1,128,f);
        if (!memchr(t->name,0,128)) fail("Invalid tensor name");
        read_exact(&t->ndim,4,1,f); read_exact(t->dims,4,4,f); read_exact(&t->count,8,1,f);
        if (t->ndim<1 || t->ndim>4 || t->count>1000000000) fail("Invalid tensor shape");
        uint64_t count=1;
        for (uint32_t j=0;j<t->ndim;j++) {
            if (!t->dims[j] || count>1000000000/t->dims[j]) fail("Tensor shape overflow");
            count*=t->dims[j];
        }
        if (count!=t->count) fail("Tensor count mismatch");
        t->data=allocate((size_t)t->count,sizeof(float)); read_exact(t->data,4,(size_t)t->count,f);
    }
    fclose(f); int C=m->channels;
    m->embedding=parameter(m,"rwkv.emb.weight",(size_t)m->vocab*C);
    m->ln0w=parameter(m,"rwkv.blocks.0.ln0.weight",C);
    m->ln0b=parameter(m,"rwkv.blocks.0.ln0.bias",C);
    m->lnw=parameter(m,"rwkv.ln_out.weight",C); m->lnb=parameter(m,"rwkv.ln_out.bias",C);
    const char *mixes[]={"att.x_r","att.x_w","att.x_k","att.x_v","att.x_a","att.x_g"};
    for (int i=0;i<m->layers;i++) {
        Layer *l=m->block+i;
#define LP(field,suffix,n) l->field=layer_param(m,i,suffix,(size_t)(n))
        LP(ln1w,"ln1.weight",C); LP(ln1b,"ln1.bias",C);
        LP(ln2w,"ln2.weight",C); LP(ln2b,"ln2.bias",C);
        for(int j=0;j<6;j++) l->mix[j]=layer_param(m,i,mixes[j],C);
        l->dw=layer_tensor(m,i,"att.w1")->dims[1];
        l->da=layer_tensor(m,i,"att.a1")->dims[1];
        l->dg=layer_tensor(m,i,"att.g1")->dims[1];
        l->dv=i?layer_tensor(m,i,"att.v1")->dims[1]:0;
        l->ff=layer_tensor(m,i,"ffn.key.weight")->dims[1];
        LP(w0,"att.w0",C); LP(w1,"att.w1",C*l->dw); LP(w2,"att.w2",C*l->dw);
        LP(a0,"att.a0",C); LP(a1,"att.a1",C*l->da); LP(a2,"att.a2",C*l->da);
        LP(g1,"att.g1",C*l->dg); LP(g2,"att.g2",C*l->dg);
        if(i) { LP(v0,"att.v0",C); LP(v1,"att.v1",C*l->dv); LP(v2,"att.v2",C*l->dv); }
        LP(kk,"att.k_k",C); LP(ka,"att.k_a",C); LP(rk,"att.r_k",C);
        LP(r,"att.receptance.weight",C*C); LP(k,"att.key.weight",C*C);
        LP(v,"att.value.weight",C*C); LP(o,"att.output.weight",C*C);
        LP(gnw,"att.ln_x.weight",C); LP(gnb,"att.ln_x.bias",C);
        LP(fmix,"ffn.x_k",C); LP(fk,"ffn.key.weight",C*l->ff); LP(fv,"ffn.value.weight",C*l->ff);
#undef LP
    }
    const char *heads[]={"cls","sts","retr"};
    for(int i=0;i<3;i++) {
        char name[128]; Head *head=m->heads+i;
#define HP(field,suffix,n) snprintf(name,sizeof(name),"head.%s_head.%s",heads[i],suffix); head->field=parameter(m,name,(size_t)(n))
        HP(fc1,"fc1.weight",C*C); HP(fc2,"fc2.weight",C*C);
        HP(nw,"norm.weight",C); HP(nb,"norm.bias",C);
#undef HP
    }
    return m;
}

API void rwkv_free(Model *m) {
    if(!m) return;
    for(int i=0;i<m->ntensors;i++) free(m->tensors[i].data);
    free(m->tensors); free(m->trie); free(m);
}
API int rwkv_channels(Model *m) { return m?m->channels:0; }
API int rwkv_layers(Model *m) { return m?m->layers:0; }

/* Small 8x32 tile: reuse each weight for eight rows. The inner loop is plain C,
   compiler-vectorizable across independent output channels (no reduction SIMD). */
static void matmul(float *out,const float *x,const float *w,int M,int K,int N) {
    #pragma omp parallel for collapse(2) schedule(static)
    for(int row=0;row<M;row+=8) for(int col=0;col<N;col+=32) {
        int nr=M-row<8?M-row:8, nc=N-col<32?N-col:32;
        float s0[32]={0},s1[32]={0},s2[32]={0},s3[32]={0};
        float s4[32]={0},s5[32]={0},s6[32]={0},s7[32]={0};
        for(int k=0;k<K;k++) {
            const float *wk=w+((size_t)(col/32)*K+k)*32;
            float a0=x[(size_t)row*K+k];
            float a1=nr>1?x[(size_t)(row+1)*K+k]:0;
            float a2=nr>2?x[(size_t)(row+2)*K+k]:0;
            float a3=nr>3?x[(size_t)(row+3)*K+k]:0;
            float a4=nr>4?x[(size_t)(row+4)*K+k]:0;
            float a5=nr>5?x[(size_t)(row+5)*K+k]:0;
            float a6=nr>6?x[(size_t)(row+6)*K+k]:0;
            float a7=nr>7?x[(size_t)(row+7)*K+k]:0;
            #if defined(__GNUC__)
            #pragma GCC unroll 32
            #endif
            for(int j=0;j<32;j++) {
                s0[j]+=a0*wk[j]; s1[j]+=a1*wk[j]; s2[j]+=a2*wk[j]; s3[j]+=a3*wk[j];
                s4[j]+=a4*wk[j]; s5[j]+=a5*wk[j]; s6[j]+=a6*wk[j]; s7[j]+=a7*wk[j];
            }
        }
        for(int j=0;j<nc;j++) {
            out[(size_t)row*N+col+j]=s0[j];
            if(nr>1) out[(size_t)(row+1)*N+col+j]=s1[j];
            if(nr>2) out[(size_t)(row+2)*N+col+j]=s2[j];
            if(nr>3) out[(size_t)(row+3)*N+col+j]=s3[j];
            if(nr>4) out[(size_t)(row+4)*N+col+j]=s4[j];
            if(nr>5) out[(size_t)(row+5)*N+col+j]=s5[j];
            if(nr>6) out[(size_t)(row+6)*N+col+j]=s6[j];
            if(nr>7) out[(size_t)(row+7)*N+col+j]=s7[j];
        }
    }
}
API void rwkv_matmul(float *out,const float *x,const float *w,int M,int K,int N) {
    float *packed=pack_matrix(w,K,N); matmul(out,x,packed,M,K,N); free(packed);
}

static void layernorm(float *out,const float *x,const float *w,const float *b,int rows,int C,float eps) {
    #pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++) {
        const float *p=x+(size_t)r*C; float mean=0,var=0;
        for(int j=0;j<C;j++) mean+=p[j];
        mean/=C;
        for(int j=0;j<C;j++) { float d=p[j]-mean; var+=d*d; }
        float scale=1.0f/sqrtf(var/C+eps);
        for(int j=0;j<C;j++) out[(size_t)r*C+j]=(p[j]-mean)*scale*w[j]+b[j];
    }
}
static float sigmoid(float x) { return 1.0f/(1.0f+expf(-x)); }
static void mix(float *out,const float *x,const float *weight,int B,int T,int C) {
    #pragma omp parallel for schedule(static)
    for(int r=0;r<B*T;r++) {
        const float *p=x+(size_t)r*C; const float *prev=r%T?p-C:NULL;
        #pragma omp simd
        for(int j=0;j<C;j++) out[(size_t)r*C+j]=p[j]+((prev?prev[j]:0)-p[j])*weight[j];
    }
}

/* RWKV-7 recurrence. Each head owns an FP32 state[key_channel,value_channel].
   a=-normalized_key, b=normalized_key*learning_rate; decay=exp(-exp(w)). */
API void rwkv_recurrence(float *out,const float *r,const float *decay,const float *k,
                        const float *v,const float *kk,const float *a,int B,int T,int C) {
    const int H=C/64;
    #pragma omp parallel for collapse(2) schedule(static)
    for(int bb=0;bb<B;bb++) for(int h=0;h<H;h++) {
        float state[64*64]={0};
        for(int t=0;t<T;t++) {
            size_t off=((size_t)bb*T+t)*C+h*64;
            const float *rt=r+off,*dt=decay+off,*kt=k+off,*vt=v+off,*nt=kk+off,*at=a+off;
            float sa[64]={0}, y[64]={0};
            for(int j=0;j<64;j++) {
                const float *s=state+j*64;
                #pragma omp simd
                for(int i=0;i<64;i++) sa[i]-=s[i]*nt[j];
            }
            for(int j=0;j<64;j++) {
                float *s=state+j*64, b=nt[j]*at[j];
                #pragma omp simd
                for(int i=0;i<64;i++) {
                    s[i]=s[i]*dt[j]+sa[i]*b+kt[j]*vt[i];
                    y[i]+=s[i]*rt[j];
                }
            }
            memcpy(out+off,y,sizeof(y));
        }
    }
}

/* Token IDs already include zero/261 left padding. pool_mask selects real EOS.
   trace, if non-null, receives [embedding, block0,...blockL-1,ln_out] FP32 arrays.
   There is no state shared across calls or sequences. */
API int rwkv_encode_tokens(Model *m,const int32_t *ids,const uint8_t *pool_mask,
                           int B,int T,int task,float *out,float *trace) {
    if(!m || !ids || !pool_mask || !out || B<1 || T<1 || B>1024 || T>131072 || task<0 || task>2) return -1;
    int C=m->channels, M=B*T; size_t n=(size_t)M*C;
    for(int r=0;r<M;r++) if(ids[r]<0 || ids[r]>=m->vocab) return -2;
    for(int b=0;b<B;b++) { int count=0; for(int t=0;t<T;t++) count+=pool_mask[b*T+t]!=0; if(!count) return -3; }
    int maxff=C,maxrank=C;
    for(int l=0;l<m->layers;l++) {
        Layer *p=m->block+l; if(p->ff>maxff) maxff=p->ff;
        int ranks[]={p->dw,p->da,p->dv,p->dg}; for(int j=0;j<4;j++) if(ranks[j]>maxrank) maxrank=ranks[j];
    }
    float *mem=allocate(n*13,sizeof(float));
    float *x=mem,*norm=x+n,*mixed=norm+n,*r=mixed+n,*w=r+n,*k=w+n,*v=k+n;
    float *a=v+n,*g=a+n,*kk=g+n,*first=kk+n,*y=first+n,*tmp=y+n;
    float *hidden=allocate((size_t)M*maxff,sizeof(float));
    float *low=allocate((size_t)M*maxrank,sizeof(float));
    for(int row=0;row<M;row++) memcpy(x+(size_t)row*C,m->embedding+(size_t)ids[row]*C,C*sizeof(float));
    if(trace) memcpy(trace,x,n*sizeof(float));
    layernorm(x,x,m->ln0w,m->ln0b,M,C,1e-5f);
    for(int li=0;li<m->layers;li++) {
        Layer *l=m->block+li;
        layernorm(norm,x,l->ln1w,l->ln1b,M,C,1e-5f);
        mix(mixed,norm,l->mix[0],B,T,C); matmul(r,mixed,l->r,M,C,C);
        mix(mixed,norm,l->mix[1],B,T,C); matmul(low,mixed,l->w1,M,C,l->dw);
        #pragma omp parallel for
        for(size_t j=0;j<(size_t)M*l->dw;j++) low[j]=tanhf(low[j]);
        matmul(w,low,l->w2,M,l->dw,C);
        #pragma omp parallel for
        for(size_t j=0;j<n;j++) {
            /* exp(-exp(-softplus(-z)-0.5)) = exp(-sigmoid(z)*exp(-0.5)). */
            w[j]=expf(-sigmoid(w[j]+l->w0[j%C])*0.6065306597126334f);
        }
        mix(mixed,norm,l->mix[2],B,T,C); matmul(k,mixed,l->k,M,C,C);
        mix(mixed,norm,l->mix[3],B,T,C); matmul(v,mixed,l->v,M,C,C);
        if(!li) memcpy(first,v,n*sizeof(float));
        else {
            matmul(low,mixed,l->v1,M,C,l->dv); matmul(tmp,low,l->v2,M,l->dv,C);
            #pragma omp parallel for
            for(size_t j=0;j<n;j++) v[j]+=(first[j]-v[j])*sigmoid(tmp[j]+l->v0[j%C]);
        }
        mix(mixed,norm,l->mix[4],B,T,C); matmul(low,mixed,l->a1,M,C,l->da);
        matmul(a,low,l->a2,M,l->da,C);
        #pragma omp parallel for
        for(size_t j=0;j<n;j++) a[j]=sigmoid(a[j]+l->a0[j%C]);
        mix(mixed,norm,l->mix[5],B,T,C); matmul(low,mixed,l->g1,M,C,l->dg);
        #pragma omp parallel for
        for(size_t j=0;j<(size_t)M*l->dg;j++) low[j]=sigmoid(low[j]);
        matmul(g,low,l->g2,M,l->dg,C);
        #pragma omp parallel for
        for(size_t off=0;off<n;off+=64) {
            float sum=0; for(int j=0;j<64;j++) { float q=k[off+j]*l->kk[(off+j)%C]; kk[off+j]=q; sum+=q*q; }
            float scale=1.0f/fmaxf(sqrtf(sum),1e-12f);
            for(int j=0;j<64;j++) { kk[off+j]*=scale; k[off+j]*=1+(a[off+j]-1)*l->ka[(off+j)%C]; }
        }
        rwkv_recurrence(y,r,w,k,v,kk,a,B,T,C);
        #pragma omp parallel for
        for(size_t off=0;off<n;off+=64) {
            float mean=0,var=0,res=0;
            for(int j=0;j<64;j++) { mean+=y[off+j]; res+=r[off+j]*k[off+j]*l->rk[(off+j)%C]; }
            mean/=64;
            for(int j=0;j<64;j++) { float d=y[off+j]-mean; var+=d*d; }
            float scale=1.0f/sqrtf(var/64+0.00064f);
            for(int j=0;j<64;j++) y[off+j]=((y[off+j]-mean)*scale*l->gnw[(off+j)%C]+l->gnb[(off+j)%C]+res*v[off+j])*g[off+j];
        }
        matmul(tmp,y,l->o,M,C,C);
        #pragma omp parallel for
        for(size_t j=0;j<n;j++) x[j]+=tmp[j];
        layernorm(norm,x,l->ln2w,l->ln2b,M,C,1e-5f);
        mix(mixed,norm,l->fmix,B,T,C); matmul(hidden,mixed,l->fk,M,C,l->ff);
        #pragma omp parallel for
        for(size_t j=0;j<(size_t)M*l->ff;j++) { float z=fmaxf(hidden[j],0); hidden[j]=z*z; }
        matmul(tmp,hidden,l->fv,M,l->ff,C);
        #pragma omp parallel for
        for(size_t j=0;j<n;j++) x[j]+=tmp[j];
        if(trace) memcpy(trace+(size_t)(li+1)*n,x,n*sizeof(float));
    }
    layernorm(x,x,m->lnw,m->lnb,M,C,1e-5f);
    if(trace) memcpy(trace+(size_t)(m->layers+1)*n,x,n*sizeof(float));
    for(int b=0;b<B;b++) {
        float *p=norm+(size_t)b*C; memset(p,0,C*sizeof(float)); int count=0;
        for(int t=0;t<T;t++) if(pool_mask[b*T+t]) {
            count++; for(int j=0;j<C;j++) p[j]+=x[((size_t)b*T+t)*C+j];
        }
        for(int j=0;j<C;j++) p[j]/=count;
    }
    Head *head=m->heads+task;
    matmul(tmp,norm,head->fc1,B,C,C);
    for(int j=0;j<B*C;j++) tmp[j]=fmaxf(tmp[j],0);
    matmul(y,tmp,head->fc2,B,C,C);
    for(int j=0;j<B*C;j++) y[j]+=norm[j];
    layernorm(out,y,head->nw,head->nb,B,C,1e-5f);
    for(int b=0;b<B;b++) {
        float sum=0; for(int j=0;j<C;j++) sum+=out[b*C+j]*out[b*C+j];
        float scale=1.0f/fmaxf(sqrtf(sum),1e-12f);
        for(int j=0;j<C;j++) out[b*C+j]*=scale;
    }
    free(low); free(hidden); free(mem); return 0;
}

/* Byte trie tokenizer; vocabulary is exported with ast.literal_eval, never eval. */
static int new_node(Model *m,unsigned char byte) {
    if(m->ntrie==m->trie_capacity) {
        m->trie_capacity=m->trie_capacity?m->trie_capacity*2:4096;
        void *p=realloc(m->trie,(size_t)m->trie_capacity*sizeof(TrieNode));
        if(!p) fail("Tokenizer allocation failed");
        m->trie=p;
    }
    int n=m->ntrie++; m->trie[n]=(TrieNode){-1,-1,-1,byte}; return n;
}
API int rwkv_load_tokenizer(Model *m,const char *path) {
    if(!m) return -1;
    FILE *f=fopen(path,"rb"); if(!f) return -1;
    uint32_t h[2]; read_exact(h,4,2,f);
    if(h[0]!=0x52564f43 || h[1]>65536) fail("Invalid tokenizer header");
    free(m->trie); m->trie=NULL; m->ntrie=m->trie_capacity=0; new_node(m,0);
    for(uint32_t i=0;i<h[1];i++) {
        uint32_t entry[2]; read_exact(entry,4,2,f);
        if(entry[0]>=65536 || !entry[1] || entry[1]>65536) fail("Invalid vocabulary entry");
        unsigned char *bytes=allocate(entry[1],1); read_exact(bytes,1,entry[1],f); int n=0;
        for(uint32_t j=0;j<entry[1];j++) {
            int child=m->trie[n].child;
            while(child>=0 && m->trie[child].byte!=bytes[j]) child=m->trie[child].sibling;
            if(child<0) { child=new_node(m,bytes[j]); m->trie[child].sibling=m->trie[n].child; m->trie[n].child=child; }
            n=child;
        }
        m->trie[n].token=entry[0]; free(bytes);
    }
    fclose(f); return 0;
}
API int rwkv_tokenize(Model *m,const char *text,int32_t *out,int capacity) {
    if(!m || !m->trie || !text || !out || capacity<0) return -1;
    const unsigned char *p=(const unsigned char*)text; size_t len=strlen(text),pos=0; int count=0;
    while(pos<len) {
        int n=0,token=-1; size_t end=pos,j=pos;
        while(j<len) {
            int child=m->trie[n].child;
            while(child>=0 && m->trie[child].byte!=p[j]) child=m->trie[child].sibling;
            if(child<0) break;
            n=child; j++; if(m->trie[n].token>=0) { token=m->trie[n].token; end=j; }
        }
        if(token<0 || count>=capacity) return -1;
        out[count++]=token; pos=end;
    }
    return count;
}

typedef struct { int32_t *ids; uint8_t *mask; int len, neos, capacity; } Sequence;
static void append(Sequence *s,int32_t id) {
    if(s->len>=s->capacity) fail("Sequence capacity exceeded");
    s->ids[s->len]=id; s->mask[s->len++]=(uint8_t)(id==EOS); s->neos+=(id==EOS);
}
static void eos_chunks(Sequence *s,const int32_t *ids,int n,int neos) {
    int step=(n+neos-1)/neos; if(step<1) step=1;
    for(int chunk=0;chunk<neos;chunk++) {
        int end=(chunk+1)*step; if(end>n) end=n;
        for(int j=chunk*step;j<end;j++) append(s,ids[j]);
        append(s,EOS);
    }
}

/* Exactly mirror custom_embedding_model._build_batch and RWKV.pad_left:
   repeat short texts, evenly split long texts, pool real EOS, pad with 0, then 261. */
API int rwkv_prepare(Model *m,const char **texts,int B,int ctx,int chunk,
                     int32_t **ids_out,uint8_t **mask_out) {
    if(!m || !texts || !ids_out || !mask_out || B<1 || B>1024 || ctx<1 || ctx>16384 || chunk<1 || chunk>ctx) return -1;
    int neos=ctx/chunk,maxlen=0,maxeos=0;
    Sequence *seq=allocate(B,sizeof(Sequence));
    for(int b=0;b<B;b++) {
        if(!texts[b]) fail("Null input text");
        size_t bytes=strlen(texts[b]);
        if(bytes>INT_MAX-1) fail("Text too long");
        int32_t *raw=allocate(bytes+1,sizeof(int32_t));
        int n=rwkv_tokenize(m,texts[b],raw,(int)bytes+1); if(n<0) fail("Tokenizer failed");
        if(n>ctx*8) n=ctx*8;
        Sequence *s=seq+b; s->capacity=ctx*16+neos*16+32;
        s->ids=allocate(s->capacity,sizeof(int32_t)); s->mask=allocate(s->capacity,1);
        if(n<=ctx-neos && n>=chunk) eos_chunks(s,raw,n,neos);
        else if(n<chunk) { for(int j=0;j<neos;j++) { for(int k=0;k<n;k++) append(s,raw[k]); append(s,EOS); } }
        else {
            int splits=(n+ctx-1)/ctx,step=(n+splits-1)/splits;
            for(int j=0;j<splits;j++) { int start=j*step,end=start+step; if(end>n) end=n; eos_chunks(s,raw+start,end-start,neos); }
        }
        free(raw); if(s->len>maxlen) maxlen=s->len; if(s->neos>maxeos) maxeos=s->neos;
    }
    /* Original wrapper takes max length BEFORE adding unused EOS; for its chunk
       shapes this still fits. Validate rather than silently truncate if it doesn't. */
    int T=(maxlen+15)/16*16,alignment=T-maxlen;
    int32_t *ids=allocate((size_t)B*T,sizeof(int32_t)); uint8_t *mask=allocate((size_t)B*T,1);
    for(int b=0;b<B;b++) {
        Sequence *s=seq+b; int extra=maxeos-s->neos,zeros=maxlen-s->len-extra;
        if(zeros<0) fail("EOS padding exceeds batch length");
        for(int j=0;j<alignment;j++) ids[b*T+j]=ALIGN_TOKEN;
        for(int j=0;j<extra;j++) ids[b*T+alignment+zeros+j]=EOS;
        memcpy(ids+b*T+T-s->len,s->ids,s->len*sizeof(int32_t));
        memcpy(mask+b*T+T-s->len,s->mask,s->len);
        free(s->ids); free(s->mask);
    }
    free(seq); *ids_out=ids; *mask_out=mask; return T;
}
API void rwkv_release(void *p) { free(p); }
API int rwkv_encode_texts(Model *m,const char **texts,int B,int ctx,int chunk,int task,float *out) {
    int32_t *ids=NULL; uint8_t *mask=NULL;
    int T=rwkv_prepare(m,texts,B,ctx,chunk,&ids,&mask); if(T<0) return T;
    int status=rwkv_encode_tokens(m,ids,mask,B,T,task,out,NULL);
    free(ids); free(mask); return status;
}

#ifndef RWKV_NO_MAIN
int main(int argc,char **argv) {
    if(argc!=4) { fprintf(stderr,"Usage: %s weights.bin tokenizer.bin \"UTF-8 text\"\n",argv[0]); return 1; }
    rwkv_threads(8); Model *m=rwkv_load(argv[1]); if(!m) return 1;
    if(rwkv_load_tokenizer(m,argv[2])) fail("Cannot load tokenizer");
    float *out=allocate(m->channels,sizeof(float)); const char *texts[]={argv[3]};
    int status=rwkv_encode_texts(m,texts,1,2048,512,2,out);
    if(!status) { printf("["); for(int j=0;j<m->channels;j++) printf("%s%.9g",j?",":"",out[j]); printf("]\n"); }
    free(out); rwkv_free(m); return status?1:0;
}
#endif
