########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
from collections import defaultdict
from typing import Dict, List, Tuple
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.strategies import DeepSpeedStrategy
from transformers import SiglipVisionModel
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam
from .dataset import EOS_INDEX, IMAGE_TOKEN_INDEX, STOP_TOKEN_INDEX
from .utils import compress_parameter_names
from .trainer import build_layerwise_lr_map

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
CHUNK_LEN = 16
flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db

def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

########################################################################################################
# RWKV TimeMix
########################################################################################################

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            if self.layer_id != 0: # not needed for the first layer
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            # D_GATE_LORA = 128
            if C != 1024:
                D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            else:
                D_GATE_LORA = 128
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()


    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

########################################################################################################
# RWKV ChannelMix
########################################################################################################
class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)

########################################################################################################
# RWKV Block
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        #self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)

    def pad_left(self, x, num_tokens_to_pad):
        # pad left with eos token embedding
        if num_tokens_to_pad != 0:
            # left padding by add eos token at the beginning
            eos_idx = torch.full(
                (x.size(0), num_tokens_to_pad),
                STOP_TOKEN_INDEX,
                dtype=torch.long,
                device=x.device,
            )
            eos_emb = self.emb(eos_idx)
            x = torch.cat((eos_emb, x), dim=1)
        return x

    def unpad(self, x, num_tokens_to_pad):
        # unpad
        if num_tokens_to_pad > 0:
            x = x[:, num_tokens_to_pad:]
        return x

    def forward(self, x):
        args = self.args

        num_tokens_to_pad = (
            CHUNK_LEN - x.size(1) % CHUNK_LEN if x.size(1) % CHUNK_LEN != 0 else 0
        )
        x = self.pad_left(x, num_tokens_to_pad)
        if args.dropout > 0:
            x = self.drop0(x)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)
        return self.unpad(x, num_tokens_to_pad)


class MLPWithContextGating(nn.Module):
    def __init__(self, in_dim, n_embd):
        super().__init__()
        self.gate = nn.Linear(in_dim, in_dim, bias=False)
        self.o_proj = nn.Linear(in_dim, n_embd, bias=False)
        self.ln_v = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x: [B, T, D]
        gating = torch.sigmoid(self.gate(x))
        return self.ln_v(self.o_proj(x * gating))


class NonlinearHead(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim_in
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim_hidden, dim_out, bias=False)
        self.norm = nn.LayerNorm(dim_out)
        # zero initialization to make the head an identity function at the beginning
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        # Residual connection: stabilize training, avoid destroying embedding geometry
        h = self.fc2(self.act(self.fc1(x)))
        return self.norm(h + x)
   

class MultiEOSPooling(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor):
        # x: [B,L,D], mask: [B,L] (bool)
        # to handle variable number of eos tokens
        # mask the value to 0 where eos_mask is False
        x = x.masked_fill(~mask.unsqueeze(-1), 0)
        return x.sum(1) / mask.sum(1, keepdim=True)    # [B,D]

    def forward(self, x: torch.Tensor, eos_mask: torch.Tensor):
        # x: [B,L,D], eos_mask: [B,L] (bool)
        return self._masked_mean(x, eos_mask)
    

class MultiTaskHead(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int):
        super().__init__()
        self.cls_head = NonlinearHead(dim_in, dim_hidden)
        self.sts_head = NonlinearHead(dim_in, dim_hidden)
        self.retr_head = NonlinearHead(dim_in, dim_hidden)
        self.task_keys = ["[CLS]", "[STS]", "[RETR]"]
        self.num_tasks = len(self.task_keys)

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor):
        # x: [B,D], task: [B] (str)
        gating_mask = F.one_hot(task_ids, num_classes=self.num_tasks).float()  # [B, 3]
        x = torch.stack([self.cls_head(x), self.sts_head(x), self.retr_head(x)], dim=1)  # [B, 3, D]
        x = (x * gating_mask.unsqueeze(-1)).sum(1)  # [B, D]
        return x


class VisualRWKVEmbed(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rwkv = RWKV(args)
        if len(args.load_model) > 0:
            self.load_rwkv_from_pretrained(args.load_model)
        self.vit = SiglipVisionModel.from_pretrained(
            args.vision_tower_path,
            attn_implementation="sdpa",
            )
        self.freeze_vit()
        self.proj = MLPWithContextGating(self.vit.config.hidden_size, args.n_embd)
        self.pool = nn.AdaptiveAvgPool2d(int(args.num_token_per_image ** 0.5))
        self.eos_pool = MultiEOSPooling()
        # projection head for different tasks
        self.head = MultiTaskHead(args.n_embd, args.n_embd)

    def load_rwkv_from_pretrained(self, path):
        self.rwkv.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
        rank_zero_info(f"Loaded pretrained RWKV from {path}")

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def freeze_vit(self):
        self.vit.requires_grad_(False)
    
    def freeze_rwkv(self, num_layers_to_freeze):
        # freeze all layers including embedding and lm head
        if num_layers_to_freeze == self.args.n_layer:
            self.rwkv.requires_grad_(False)
        # otherwise, freeze only the first num_layers_to_freeze layers
        for i, block in enumerate(self.rwkv.blocks):
            if i < num_layers_to_freeze:
                for p in block.parameters():
                    p.requires_grad_(False)
            else:
                for p in block.parameters():
                    p.requires_grad_(True)

    def freeze_emb(self):
        self.rwkv.emb.requires_grad_(False)

    def freeze_proj(self):
        self.proj.requires_grad_(False)

    def configure_optimizers(self):
        name_of_trainable_params = [n for n, p in self.named_parameters() if p.requires_grad]
        compressed_name_of_trainable_params = compress_parameter_names(name_of_trainable_params)
        rank_zero_info(f"Name of trainable parameters in optimizers: {compressed_name_of_trainable_params}")
        rank_zero_info(f"Number of trainable parameters in optimizers: {len(name_of_trainable_params)}")
        # build layer-wise learning rate map (LLRD)
        layer_lr_map, layer_lr_summary = build_layerwise_lr_map(self, self.args)
        if layer_lr_summary:
            lr_log = ", ".join(f"{name}: {lr:.7e}" for name, lr in layer_lr_summary)
            rank_zero_info(f"Layer-wise learning rates: {lr_log}")

        param_groups = defaultdict(list)
        num_weight_decay_params = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.squeeze().shape) < 2:
                weight_decay = 0.0
            else:
                weight_decay = self.args.weight_decay if self.args.weight_decay > 0 else 0.0
                if weight_decay > 0:
                    num_weight_decay_params += 1
            lr = layer_lr_map.get(name, self.args.lr_init)
            param_groups[(weight_decay, lr)].append(param)

        if self.args.weight_decay > 0 and num_weight_decay_params > 0:
            rank_zero_info(
                f"Number of parameters with weight decay: {num_weight_decay_params}, with value: {self.args.weight_decay}"
            )

        optim_groups = []
        for (weight_decay, lr), params in sorted(param_groups.items(), key=lambda item: (item[0][1], item[0][0])):
            group = {"params": params, "weight_decay": weight_decay, "lr": lr, "initial_lr": lr}
            optim_groups.append(group)

        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
    
    def preparing_embedding_fused(self, samples):
        raise NotImplementedError("Fused embedding is not implemented yet.")

    def preparing_embedding_text_only(self, samples):
        '''
        mask used to select the output feature after rwkv
        '''
        query_ids_embeds = self.rwkv.emb(samples["query_ids"])
        query_mask = samples["query_ids"] == EOS_INDEX

        positive_ids_embeds = self.rwkv.emb(samples["positive_ids"])
        positive_mask = samples["positive_ids"] == EOS_INDEX

        negative_embeds: List[torch.Tensor] = []
        negative_masks: List[torch.Tensor] = []
        negative_keys: List[str] = []
        neg_idx = 1
        while True:
            neg_key = f"negative{neg_idx}_ids"
            if neg_key not in samples:
                break
            negative_embeds.append(self.rwkv.emb(samples[neg_key]))
            negative_masks.append(samples[neg_key] == EOS_INDEX)
            negative_keys.append(f"negative{neg_idx}")
            neg_idx += 1

        return (
            query_ids_embeds,
            query_mask,
            positive_ids_embeds,
            positive_mask,
            negative_embeds,
            negative_masks,
            negative_keys,
        )

    def forward(self, samples):
        if "images" in samples:
            raise NotImplementedError("Multi-Modal embedding is not implemented yet.")
        else:
            (
                query_emb,
                query_mask,
                positive_emb,
                positive_mask,
                negative_embeds,
                negative_masks,
                negative_keys,
            ) = self.preparing_embedding_text_only(samples)

        B, L, D = query_emb.shape
        # get the features from rwkv
        embeddings: List[torch.Tensor] = [query_emb, positive_emb]
        embeddings.extend(negative_embeds)
        concat_input = torch.cat(embeddings, dim=0)
        res = self.rwkv(concat_input)
        splits = torch.split(res, B, dim=0)
        query_res = splits[0]
        positive_res = splits[1]
        negative_res_list = list(splits[2:])

        query_feature = query_res[query_mask].view(B, -1, D)
        query_feature = self.eos_pool(query_feature, samples['query_eos_mask'])  # -> [B,D]
        query_feature = self.head(query_feature, samples['task'])  # -> [B,D]

        positive_feature = positive_res[positive_mask].view(B, -1, D)
        positive_feature = self.eos_pool(positive_feature, samples['positive_eos_mask'])
        positive_feature = self.head(positive_feature, samples['task'])  # -> [B,D]

        negative_features: List[torch.Tensor] = []
        for neg_res, neg_mask, neg_key in zip(negative_res_list, negative_masks, negative_keys):
            neg_feature = neg_res[neg_mask].view(B, -1, D)
            neg_feature = self.eos_pool(neg_feature, samples[f"{neg_key}_eos_mask"])
            neg_feature = self.head(neg_feature, samples['task'])  # -> [B,D]
            negative_features.append(neg_feature)

        return query_feature, positive_feature, negative_features

    def training_step(self, batch):
        query_vec, pos_vec, neg_vecs = self(batch)   # [B, D], [B, D], List[B, D]
        B, D = query_vec.shape
        query_vec = F.normalize(query_vec.float(), dim=1)
        pos_vec = F.normalize(pos_vec.float(), dim=1)
        neg_vecs = [F.normalize(neg_vec.float(), dim=-1) for neg_vec in neg_vecs]
        query_src_ids: torch.Tensor = batch['source'].long()

        candidates = torch.cat([pos_vec] + neg_vecs, dim=0)  # [N, D], N = (1 + num_neg) * B
        candidate_src_ids = torch.cat([query_src_ids] + [query_src_ids]*len(neg_vecs), dim=0)  # [N]
        logits = query_vec @ candidates.t()  # [B, N]
        targets = torch.arange(B, device=logits.device)

        logits = logits / self.args.temperature
        # mask out the logits from different sources, avoid cross-source influence
        logits_mask = (query_src_ids.view(-1, 1) == candidate_src_ids.view(1, -1))  # [B,N] bool
        logits = logits.masked_fill(~logits_mask, -1e9)
        loss = F.cross_entropy(logits, targets, reduction='none')  # [B]
        loss = (loss * batch['weight']).mean() if 'weight' in batch else loss.mean()
        return loss

    def training_step_end(self, batch_parts):
        if pl.__version__[0] != '2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

    def adaptive_pooling(self, image_features):
        B, L, D = image_features.shape
        H_or_W = int(L**0.5)
        image_features = image_features.view(B, H_or_W, H_or_W, D).permute(0, 3, 1, 2)
        image_features = self.pool(image_features).view(B, D, -1).permute(0, 2, 1)
        return image_features
    
    def encode_images(self, images):
        # print('in encode: ', images.shape)
        B, N, C, H, W = images.shape
        
        images = images.view(B*N, C, H, W)
        image_features = self.vit(images).last_hidden_state
        L, D = image_features.shape[1], image_features.shape[2]
        # rerange [B*N, L, D] -> [B, N, L, D]
        image_features = image_features.view(B, N, L, D)[:, 0, :, :]
        image_features = self.adaptive_pooling(image_features)
        return self.proj(image_features)