########################################################################################################
# The EmbeddingRWKV Model - https://github.com/add_later
########################################################################################################

from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import SiglipVisionModel, SiglipVisionConfig

from .reference.rwkv7 import RWKV_x070, DTYPE


Tokens = List[int]
BatchTokens = Sequence[Sequence[int]]


def generate_eos_mask(tokens: Union[Tokens, BatchTokens], eos_token_id: int) -> torch.Tensor:
    if isinstance(tokens, Sequence) and len(tokens) > 0 and isinstance(tokens[0], Sequence):
        # batch of sequences
        batch_size = len(tokens)
        max_len = max(len(t) for t in tokens)
        eos_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        for i, t in enumerate(tokens):
            for j, token_id in enumerate(t):
                if token_id == eos_token_id:
                    eos_mask[i, j] = True
        return eos_mask
    else:
        # single sequence
        seq_len = len(tokens)
        eos_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        for j, token_id in enumerate(tokens):
            if token_id == eos_token_id:
                eos_mask[0, j] = True
        return eos_mask

def extract_module_state_dict(state_dict, module_name: str):
    extracted = {}
    for key in state_dict:
        if key.startswith(module_name + "."):
            new_key = key[len(module_name) + 1 :]
            extracted[new_key] = state_dict[key]
    return extracted


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

    def forward(self, x: torch.Tensor, task_type: str):
        # x: [B,D], task: str
        if task_type == "[CLS]":
            x = self.cls_head(x)
        elif task_type == "[STS]":
            x = self.sts_head(x)
        elif task_type == "[RETR]":
            x = self.retr_head(x)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        return x


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
    

class EmbeddingRWKV(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        state_dict = torch.load(model_path, map_location='cpu', mmap=True, weights_only=True)
        rwkv_dict = extract_module_state_dict(state_dict, "rwkv")
        head_dict = extract_module_state_dict(state_dict, "head")
        vit_dict = extract_module_state_dict(state_dict, "vit")
        proj_dict = extract_module_state_dict(state_dict, "proj")
        # must before init rwkv, otherwise n_head and head_size will be flattened
        self.n_head, self.head_size = rwkv_dict['blocks.0.att.r_k'].shape
        # prepare rwkv
        model_prefix = model_path[:-4] if model_path.endswith(".pth") else model_path
        args = SimpleNamespace(MODEL_NAME=model_prefix)
        self.rwkv = RWKV_x070(args, rwkv_dict)
        self.device = self.rwkv.z['emb.weight'].device
        self.dtype = self.rwkv.z['emb.weight'].dtype
        # prepare head
        self.n_embd = self.n_head * self.head_size
        self.head = MultiTaskHead(self.n_embd, self.n_embd)
        self.head.load_state_dict(head_dict)
        self.head.to(self.dtype).to(self.device)
        self.eos_pool = MultiEOSPooling()
        self.eos_token_id = 65535
        self.num_token_per_image = 256
        self.has_vision = len(vit_dict) > 0
        self.vit = None
        self.proj = None
        self.pool = None

        if self.has_vision:
            vision_tower_path = "google/siglip2-base-patch16-256"
            config = SiglipVisionConfig.from_pretrained(
                vision_tower_path, 
                attn_implementation="sdpa"
            )
            self.vit = SiglipVisionModel(config)
            self.vit.load_state_dict(vit_dict, strict=False)
            self.vit.to(self.device)
            self.vit.eval()
            self.vit.requires_grad_(False)

            self.proj = MLPWithContextGating(self.vit.config.hidden_size, self.n_embd)
            if len(proj_dict) > 0:
                self.proj.load_state_dict(proj_dict, strict=False)
            self.proj.to(self.dtype).to(self.device)
            self.pool = nn.AdaptiveAvgPool2d(int(self.num_token_per_image ** 0.5))

    def generate_zero_state(self, bsz):
        return self.rwkv.generate_zero_state(bsz)

    def adaptive_pooling(self, image_features: torch.Tensor):
        B, L, D = image_features.shape
        H_or_W = int(L**0.5)
        image_features = image_features.view(B, H_or_W, H_or_W, D).permute(0, 3, 1, 2)
        image_features = self.pool(image_features).view(B, D, -1).permute(0, 2, 1)
        return image_features

    def encode_images(self, images: torch.Tensor):
        image_features = self.vit(images).last_hidden_state
        if image_features.shape[1] != self.num_token_per_image:
            image_features = self.adaptive_pooling(image_features)
        return self.proj(image_features.to(self.dtype))

    @torch.inference_mode()
    def forward(
        self,
        tokens: Union[Tokens, BatchTokens] = None,
        state: Optional[List[torch.Tensor]] = None,
        full_output: bool = True,
        task_type: str = None,
        images: Optional[torch.Tensor] = None,
    ):
        '''
        tokens can be None (for image-only input)
        images can be None (for text-only input)
        1. If both tokens and images are provided, they will be concatenated as:
           [text tokens] + [image tokens] + [eos token]
        2. If only tokens are provided, process as text-only input.
        3. If only images are provided, process as image-only input:
           [image tokens] + [eos token]
        4. If neither tokens nor images are provided, raise an error.
        '''
        if tokens is None and images is None:
            raise ValueError("At least one of tokens or images must be provided.")
        if tokens is not None and images is not None:
            return self.forward_multimodal(tokens, images, state, full_output, task_type)
        if tokens is not None and images is None:
            return self.forward_text_only(tokens, state, full_output, task_type)
        if tokens is None and images is not None:
            return self.forward_image_only(images, state, full_output, task_type)
        
    def forward_image_only(
        self,
        images: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
        full_output: bool = False,
        task_type: str = None,
    ):
        if not self.has_vision:
            raise ValueError("Vision encoder not found in checkpoint; cannot process images.")
        if isinstance(images, torch.Tensor) and images.dim() == 3:
            images = images.unsqueeze(0) # [C,H,W] -> [1,C,H,W]
        image_embeds = self.encode_images(images.to(self.device))
        B = image_embeds.size(0)
        # append eos token embedding
        image_eos_idx = torch.full((B, 1), self.eos_token_id, device=self.device, dtype=torch.long)
        image_eos_emb = self.rwkv.z['emb.weight'][image_eos_idx]  # [B, 1, D]
        image_embeds = torch.cat([image_embeds, image_eos_emb], dim=1)  # [B, N+1, D]
        if state is None:
            B = images.size(0)
            state = self.generate_zero_state(B)
        output = self.rwkv.forward_one_batch_alt(image_embeds, state, full_output)
        if full_output is True:
            feature = output[:,-1,:] # [B,D]
        else:
            feature = output # last token [B,D]
        # apply head
        if task_type is None:
            task_type = "[RETR]"
            embedding = self.head(feature, task_type)
        else:
            embedding = self.head(feature, task_type)

        return embedding, state

    def forward_text_only(
        self,
        tokens: Union[Tokens, BatchTokens],
        state: Optional[List[torch.Tensor]] = None,
        full_output: bool = True,
        task_type: str = None,
    ):
        is_batch = isinstance(tokens, Sequence) and len(tokens) > 0 and isinstance(tokens[0], Sequence)
        if state is None:
            B = len(tokens) if is_batch else 0
            state = self.generate_zero_state(B)

        if is_batch:
            output = self.rwkv.forward_batch(tokens, state, full_output)
            if full_output is True: # else it's just the last token
                output = torch.stack(output)
        else:
            output = self.rwkv.forward(tokens, state, full_output)
        # prepare eos mask and pooling
        if full_output is True:
            eos_mask = generate_eos_mask(tokens, self.eos_token_id).to(self.device)
            feature = self.eos_pool(output, eos_mask)
        else:
            feature = output
        # apply head
        if task_type is None:
            task_type = "[RETR]"
            embedding = self.head(feature, task_type)
        else:
            embedding = self.head(feature, task_type)

        return embedding, state

    def forward_multimodal(
        self,
        tokens: Union[Tokens, BatchTokens],
        images: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
        full_output: bool = True,
        task_type: str = None,
    ):
        if not self.has_vision:
            raise ValueError("Vision encoder not found in checkpoint; cannot process images.")
        # 1. ensure text and image are paired in batch
        is_batch = isinstance(tokens, Sequence) and len(tokens) > 0 and isinstance(tokens[0], Sequence)
        if not is_batch:
            tokens = [tokens] # batch
        if images.dim() == 3:
            images = images.unsqueeze(0) # [C,H,W] -> [1,C,H,W]
        assert len(tokens) == images.size(0), f"Batch size mismatch between tokens and images: {len(tokens)} vs {images.size(0)}"

        B = len(tokens)
        
        # 2. encode images
        image_embeds = self.encode_images(images.to(self.device))
        # append eos token embedding
        image_eos_idx = torch.full((B, 1), self.eos_token_id, device=self.device, dtype=torch.long)
        image_eos_emb = self.rwkv.z['emb.weight'][image_eos_idx]  # [B, 1, D]
        image_embeds = torch.cat([image_embeds, image_eos_emb], dim=1)  # [B, N+1, D]
        # 3. prepare text embeddings
        text_embeds = self.rwkv.z['emb.weight'][torch.tensor(tokens, device=self.device)]
        # 4. concatenate text and image embeddings
        combined_embeds = torch.cat([text_embeds, image_embeds], dim=1)  # [B, L+N+1, D]
        # 5. prepare state
        if state is None:
            state = self.generate_zero_state(B)
        # 6. forward through rwkv
        output = self.rwkv.forward_one_batch_alt(combined_embeds, state, full_output)
        if full_output is True:
            text_eos_mask = generate_eos_mask(tokens, self.eos_token_id).to(self.device)
            # extend eos_mask to include image eos token
            image_eos_mask = torch.ones((B, image_embeds.size(1)), dtype=torch.bool, device=self.device)
            eos_mask = torch.cat([text_eos_mask, image_eos_mask], dim=1)  # [B, L+N+1]
            feature = self.eos_pool(output, eos_mask)
        else:
            feature = output
        # 7. apply head
        if task_type is None:
            task_type = "[RETR]"
            embedding = self.head(feature, task_type)
        else:
            embedding = self.head(feature, task_type)
        return embedding, state
        
##########################################################################################################
# RWKV Low-Level Operator
########################################################################################################### 
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
# RWKV-ReRanker Model
#########################################################################################################

class ReRanker(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(1, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Sequential(
            nn.Linear(args.n_embd, args.n_embd),
            nn.Tanh(),
            nn.Linear(args.n_embd, 1, bias=False),
        )

    def forward(self, states):
        L, B, H, S, S = states.shape
        ids = torch.zeros((B, 1), device=states.device, dtype=torch.long)
        x = self.emb(ids)  # [B, 1, D]

        v_first = torch.empty_like(x)
        for i, block in enumerate(self.blocks):
            state = states[i]  # [B, H, S, S]
            x, v_first = block(x, v_first, state)

        x = self.ln_out(x)
        return self.head(x).squeeze(-1)  # [B, 1]


class RWKVReRanker(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        state_dict = torch.load(model_path, map_location='cpu', mmap=True, weights_only=True)
        reranker_dict = extract_module_state_dict(state_dict, "reranker")
        if "token.weight" in reranker_dict:
            reranker_dict["emb.weight"] = reranker_dict.pop("token.weight")
        self.reranker_layer_idx = None
        if "reranker_layer_idx" in state_dict:
            self.reranker_layer_idx = state_dict["reranker_layer_idx"]
        # parse args
        self.n_head, self.head_size = reranker_dict['blocks.0.att.r_k'].shape
        self.n_embd = self.n_head * self.head_size
        num_reranker_layers = max(int(k.split('.')[1]) for k in reranker_dict if k.startswith('blocks.')) + 1
        args = SimpleNamespace(
            n_embd=self.n_embd,
            n_layer=num_reranker_layers,
            head_size_a=self.head_size,
            dim_att=self.n_embd,
            head_size_divisor=8,
        )
        # ranker
        self.reranker = ReRanker(args)
        self.reranker.load_state_dict(reranker_dict)
        self.reranker.to(DTYPE).to('cuda')

    @torch.inference_mode()
    def forward(self, states: torch.Tensor):
        # states: [L, B, H, S, S]
        if self.reranker_layer_idx is not None:
            selected = [states[i] for i in self.reranker_layer_idx]
            states = torch.stack(selected, dim=0)
        return self.reranker(states).squeeze(-1)  # [B]
