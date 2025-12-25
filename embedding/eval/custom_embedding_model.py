import os
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List
from tqdm import tqdm

import torch
from transformers import AutoImageProcessor
from mteb.encoder_interface import PromptType

# make sure ``src`` is on the import path when used as a package
FILE_PATH = Path(__file__).resolve()
sys.path.append(str(FILE_PATH.parent))

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from src.model import VisualRWKVEmbed

# Token constants copied from evaluate_all.py
IMAGE_TOKEN_INDEX = 65534
EOS_INDEX = 65535
PAD_INDEX = 0


@dataclass
class VisualRWKVMTEBConfig:
    """Configuration for ``VisualRWKVMTEBModel``.

    Parameters mirror the arguments used by ``evaluate_all.py`` when
    instantiating ``VisualRWKVEmbed`` so that a checkpoint can be loaded.
    """

    model_path: str
    vision_tower_path: str
    ctx_len: int = 1024
    eos_chunk_size: int | None = None
    num_token_per_image: int = 16
    n_layer: int = 12
    vocab_size: int = 65536
    n_embd: int = 768
    dim_att: int = 0
    dim_ffn: int = 0
    pre_ffn: int = 0
    head_size_a: int = 64
    head_size_divisor: int = 8
    dropout: float = 0.0
    grad_cp: int = 0
    proj_type: str = "mlp"
    load_model: str = ""
    def __post_init__(self):
        if self.dim_ffn == 0:
            self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size
        if self.dim_att == 0:
            self.dim_att = self.n_embd


class VisualRWKVMTEBModel:
    """Minimal wrapper so VisualRWKV embeds can be evaluated with MTEB."""

    def __init__(self, cfg: VisualRWKVMTEBConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.task2instruct = {}
        self.task2tag = {}
        # multi-task head
        self.task_keys = ["[CLS]", "[STS]", "[RETR]"]
        self.num_tasks = len(self.task_keys)
        self.task2id = {k: i for i, k in enumerate(self.task_keys)}

        # the RWKV implementation reads model hyper-parameters from env vars
        os.environ["RWKV_HEAD_SIZE_A"] = str(cfg.head_size_a)
        os.environ["RWKV_CTXLEN"] = str(cfg.ctx_len)

        # build args object expected by ``VisualRWKVEmbed``
        self.args = type("Args", (), cfg.__dict__)()

        self.tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
        self.image_processor = AutoImageProcessor.from_pretrained(
            cfg.vision_tower_path, use_fast=True
        )

        self.model = VisualRWKVEmbed(self.args)
        state = torch.load(cfg.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=False)
        self.model = self.model.to(self.device).bfloat16().eval()
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.model = torch.nn.DataParallel(self.model)

    # ------------------------------------------------------------------
    # helper utilities
    def _normal_ctx_process(self, ids: List[int], n_eos: int):
        '''processing for a context length text'''
        t_len = len(ids)
        input_ids, eos_mask = [], []
        real_chunk_size = max(1, math.ceil(t_len / n_eos))
        for chunk_idx in range(n_eos):
            start = chunk_idx * real_chunk_size
            end = min(start + real_chunk_size, t_len)
            chunk = ids[start:end]
            input_ids.extend(chunk)
            input_ids.append(EOS_INDEX)
            eos_mask.append(1)
        return input_ids, eos_mask

    def _short_ctx_process(self, ids: List[int], n_eos: int):
        '''processing for a short context length text, repeat the text to fill n_eos'''
        input_ids, eos_mask = [], []
        for _ in range(n_eos):
            input_ids.extend(ids)
            input_ids.append(EOS_INDEX)
            eos_mask.append(1)
        return input_ids, eos_mask

    def _long_ctx_process(self, ids: List[int], ctx_len: int, n_eos: int):
        '''processing for a long context length text, split the text'''
        t_len = len(ids)
        n_split = math.ceil(t_len / ctx_len) # 1.1 -> 2
        long_input_ids, long_eos_mask = [], []
        real_split_size = math.ceil(t_len / n_split)
        for split_idx in range(n_split):
            start = split_idx * real_split_size
            end = min(start + real_split_size, t_len)
            split_ids = ids[start:end]
            input_ids, eos_mask = self._normal_ctx_process(split_ids, n_eos)
            long_input_ids.extend(input_ids)
            long_eos_mask.extend(eos_mask)
        return long_input_ids, long_eos_mask

    def _build_batch(self, texts: List[str]):
        all_ids = [list(self.tokenizer.encode(t)) for t in texts]
        all_ids = [ids[: self.cfg.ctx_len*8] for ids in all_ids]  # truncate to ctx_len*8
        all_ids_with_eos = []
        all_eos_mask = []

        chunk_size = self.cfg.eos_chunk_size
        n_eos = self.cfg.ctx_len // chunk_size
        # process each text case by case
        for ids in all_ids:
            max_allowed_len = self.cfg.ctx_len - n_eos
            ids_len = len(ids)
            # case 1: len(ids) <= max_allowed_len and len(ids) >= chunk_size
            if ids_len <= max_allowed_len and ids_len >= chunk_size:
                input_ids, eos_mask = self._normal_ctx_process(ids, n_eos)
                all_ids_with_eos.append(input_ids)
                all_eos_mask.append(eos_mask)
            elif ids_len < chunk_size:
                # case 2: len(ids) < chunk_size
                input_ids, eos_mask = self._short_ctx_process(ids, n_eos)
                all_ids_with_eos.append(input_ids)
                all_eos_mask.append(eos_mask)
            else:
                # case 3: len(ids) > max_allowed_len
                input_ids, eos_mask = self._long_ctx_process(ids, self.cfg.ctx_len, n_eos)
                all_ids_with_eos.append(input_ids)
                all_eos_mask.append(eos_mask)

        # left pad to the longest sequence in the batch
        max_real_len = max(len(ids) for ids in all_ids_with_eos)
        max_n_eos = max(len(mask) for mask in all_eos_mask)
        batch_ids, batch_eos_mask = [], []
        for ids, mask in zip(all_ids_with_eos, all_eos_mask):
            eos_to_pad = max_n_eos - len(mask)
            if eos_to_pad > 0:
                ids = [EOS_INDEX] * eos_to_pad + ids
                mask = [0] * eos_to_pad + mask
            # left pad to max_real_len for input_ids
            pad_len = max_real_len - len(ids)
            if pad_len > 0:
                ids = [PAD_INDEX] * pad_len + ids
            batch_ids.append(ids)
            batch_eos_mask.append(mask)

        token_tensor = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        eos_mask_tensor = torch.tensor(batch_eos_mask, dtype=torch.bool, device=self.device)
        return {"query_ids": token_tensor, "query_eos_mask": eos_mask_tensor}

    # ------------------------------------------------------------------
    # MTEB API
    def encode_queries(
        self,
        queries: List[str],
        task_name: str,
        batch_size: int = 32,
        **kwargs,
    ):
        if task_name in self.task2instruct:
            instruct = self.task2instruct[task_name]
            queries = [instruct.format(query=q) for q in queries]
        return self.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(
        self,
        corpus,
        batch_size: int = 32,
        **kwargs,
    ):
        """Encode a corpus supplied by MTEB.
        """

        texts: List[str] = []
        for doc in corpus:
            if isinstance(doc, dict):
                title = doc.get("title") or ""
                text = doc.get("text") or ""
                texts.append((title + " " + text).strip())
            else:
                texts.append(doc)

        return self.encode(texts, batch_size=batch_size, **kwargs)

    def encode(
        self,
        sentences: List[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        if prompt_type == PromptType.query:
            instruct = self.task2instruct[task_name]
            if instruct and "Instruct:" not in sentences[0]:
                # avoid double instruction
                sentences = [instruct.format(query=s) for s in sentences]
        # for non-retrieval tasks, prompt type is None, add instruction to help model
        if self.task2tag[task_name] != "[RETR]":
            instruct = self.task2instruct[task_name]
            if instruct and "Instruct:" not in sentences[0]:
                # avoid double instruction
                sentences = [instruct.format(query=s) for s in sentences]
        
        embeddings = []
        batch_size = batch_size * self.gpu_count if self.gpu_count > 1 else batch_size
        for i in tqdm(range(0, len(sentences), batch_size), desc=task_name):
            batch_texts = sentences[i : i + batch_size]
            batch = self._build_batch(batch_texts)
            task_list = [self.task2tag[task_name]] * len(batch_texts)
            batch['task'] = torch.tensor([self.task2id[t] for t in task_list])
            with torch.inference_mode():
                vec, _ = self.model(batch)
                if normalize_embeddings:
                    vec = vec / vec.norm(dim=1, keepdim=True)
            embeddings.append(vec.to(torch.float32).cpu())
        return torch.cat(embeddings, dim=0).numpy()

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between query and corpus embeddings.
        """

        if isinstance(a, torch.Tensor):
            a_tensor = a.to(self.device).to(dtype=torch.float32)
        else:
            a_tensor = torch.tensor(a, dtype=torch.float32, device=self.device)

        if isinstance(b, torch.Tensor):
            b_tensor = b.to(self.device).to(dtype=torch.float32)
        else:
            b_tensor = torch.tensor(b, dtype=torch.float32, device=self.device)

        if len(a_tensor.shape) == 1:
            a_tensor = a_tensor.reshape(1, *a_tensor.shape)
        if len(b_tensor.shape) == 1:
            b_tensor = b_tensor.reshape(1, *b_tensor.shape)

        scores = a_tensor @ b_tensor.transpose(0, 1)
        return scores