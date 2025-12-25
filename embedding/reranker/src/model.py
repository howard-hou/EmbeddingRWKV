########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib, sys
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
from pathlib import Path
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam
from .dataset import EOS_INDEX, IMAGE_TOKEN_INDEX, STOP_TOKEN_INDEX
from .utils import compress_parameter_names, resolve_reranker_layer_indices
from .trainer import build_layerwise_lr_map

REFERENCE_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(REFERENCE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_PACKAGE_ROOT))

from reference.rwkv7 import RWKV_x070  # noqa: E402
PRECISIONS_TO_DTYPE = {"fp32": torch.float32, "tf32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

HEAD_SIZE = 64
def RWKV7_OP(r, w, k, v, a, b, state):
    data_type = r.dtype
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    state = state.float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)

    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)

    return out.view(B, T, C).to(dtype=data_type), state.to(dtype=data_type)

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


    def forward(self, x, v_first, state):
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

        x, state = RWKV7_OP(r, w, k, v, -kk, kk*a, state)
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
        
    def forward(self, x, v_first, state):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first, state)
        x = x + x_attn
        x = x + self.ffn(self.ln2(x))
        return x, v_first

########################################################################################################
# ReRanker Model
#########################################################################################################

class ReRanker(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layer_indices = args.reranker_layer_idx
        self.num_reranker_layers = len(self.layer_indices)
        self.emb = nn.Embedding(1, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(self.num_reranker_layers)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Sequential(
            nn.Linear(args.n_embd, args.n_embd),
            nn.Tanh(),
            nn.Linear(args.n_embd, 1, bias=False),
        )

    def forward(self, states):
        use_shared_state = getattr(self.args, "use_shared_state", False)
        if not use_shared_state:
            selected = [states[i] for i in self.layer_indices]
            states = torch.stack(selected, dim=0)
        L, B, H, S, S = states.shape
        ids = torch.zeros((B, 1), device=states.device, dtype=torch.long)
        x = self.emb(ids)  # [B, 1, D]

        v_first = torch.empty_like(x)
        for i, block in enumerate(self.blocks):
            state = states[-1] if use_shared_state else states[i]  # [B, H, S, S]
            if self.args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first, state)
            else:
                x, v_first = block(x, v_first, state)

        x = self.ln_out(x)
        return self.head(x).squeeze(-1)  # [B, 1]


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


class RWKVReRanker(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rwkv = RWKV_x070(args)
        self.rwkv_weights_on_gpu = False
        if args.vision_tower_path:
            self.vit = SiglipVisionModel.from_pretrained(
                args.vision_tower_path,
                attn_implementation="sdpa",
                )
            self.freeze_vit()
            self.proj = MLPWithContextGating(self.vit.config.hidden_size, args.n_embd)
            self.pool = nn.AdaptiveAvgPool2d(int(args.num_token_per_image ** 0.5))
        # ranker
        self.reranker = ReRanker(args)
        self.data_type = PRECISIONS_TO_DTYPE[args.precision]

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def freeze_vit(self):
        self.vit.requires_grad_(False)
    
    def freeze_reranker(self):
        # otherwise, freeze only the first num_layers_to_freeze layers
        for i, block in enumerate(self.reranker.blocks):
            for p in block.parameters():
                p.requires_grad_(False)

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

    def ids_tensor_to_list(self, ids_tensor):
        ids_list = []
        ids_np = ids_tensor.cpu().numpy()
        for ids in ids_np:
            ids_list.append(ids.tolist())
        return ids_list
    
    def forward_full(self, token_ids):
        B = len(token_ids)
        state = self.rwkv.generate_zero_state(B, self.device)
        self.rwkv.forward_batch(token_ids, state)
        wkv_state = state[1].clone()  # [L, B, H, S, S]
        return wkv_state
        
    def forward(self, samples):
        B = samples["query_ids"].shape[0]
        #query_ids = self.ids_tensor_to_list(samples["query_ids"])
        positive_ids = self.ids_tensor_to_list(samples["positive_ids"])
        neg_keys = [k for k in samples if k.startswith("negative") and k.endswith("_ids")]
        neg_ids = {neg_key: self.ids_tensor_to_list(samples[neg_key]) for neg_key in neg_keys}
        # get the states from rwkv
        # query and pos concat in dataset
        #query_state = self.forward_full(query_ids).to(self.data_type)
        positive_state = self.forward_full(positive_ids).to(self.data_type)
        neg_states = []
        for neg_key in neg_keys:
            neg_state = self.forward_full(neg_ids[neg_key]).to(self.data_type)
            neg_states.append(neg_state)
        return positive_state, neg_states
    
    def compute_logit_by_state(self, samples):
        pos_state, neg_states = self(samples)
        # pos_state: [L, B, H, S, S]
        # neg_states: list of [L, B, H, S, S], length N
        # cat to a large batch
        all_states = torch.cat([pos_state] + neg_states, dim=1)  # [L, B*(N+1), H, S, S]
        # one forward
        all_logits = self.reranker(all_states)  # [B*(N+1), 1]
        # back to pos and neg
        B = pos_state.size(1)
        pos_logits = all_logits[:B]              # [B, 1]
        neg_logits = all_logits[B:].view(len(neg_states), B).transpose(0, 1)  # [B, N]
        return pos_logits, neg_logits

    def move_rwkv_weights_to_device(self):
        device = self.device
        new_z = {}
        for k, v in self.rwkv.z.items():
            new_z[k] = v.to(device, non_blocking=True)
        self.rwkv.z = new_z
        self.rwkv_weights_on_gpu = True

    def training_step(self, batch):
        if not self.rwkv_weights_on_gpu:
            self.move_rwkv_weights_to_device()
        pos_logits, neg_logits = self.compute_logit_by_state(batch)

        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [B, N+1]
        loss = F.binary_cross_entropy_with_logits(
            logits,
            torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=1),
        )
        return loss

    def training_step_end(self, batch_parts):
        if pl.__version__[0] != '2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

    def predict(self, batch_ids: torch.Tensor) -> torch.Tensor:
        if not self.rwkv_weights_on_gpu:
            self.move_rwkv_weights_to_device()
        state = self.forward_full(self.ids_tensor_to_list(batch_ids))
        x = self.reranker(state) # [B, 1]
        return x.squeeze(-1)  # [B]