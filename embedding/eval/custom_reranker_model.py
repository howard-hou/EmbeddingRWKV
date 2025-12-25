import os
import sys
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List
from tqdm import tqdm

import torch
import torch.nn.functional as F
from mteb.encoder_interface import PromptType

# make sure ``src`` is on the import path when used as a package
FILE_PATH = Path(__file__).resolve()
sys.path.append(str(FILE_PATH.parent))

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from src.reranker import RankingHead, RWKVReranker
from src.reference.rwkv7 import RWKV_x070

# Token constants copied from evaluate_all.py
IMAGE_TOKEN_INDEX = 65534
EOS_INDEX = 65535
PAD_INDEX = 0


@dataclass
class RWKVRerankerMTEBConfig:
    """Configuration for ``RWKVRerankerMTEBModel``.

    Parameters mirror the arguments used by ``evaluate_all.py`` when
    instantiating ``RWKVEmbed`` so that a checkpoint can be loaded.
    """

    rwkv_path: str
    reranker_path: str
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
    state_reranker_dim: int = 256
    state_reranker_layers: int = 3
    def __post_init__(self):
        if self.dim_ffn == 0:
            self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size
        if self.dim_att == 0:
            self.dim_att = self.n_embd
        self.MODEL_NAME = str(Path(self.rwkv_path).resolve()).strip('.pth')


class CustomRerankerModel:
    """Minimal wrapper so RWKV embeds can be evaluated with MTEB."""

    def __init__(self, cfg: RWKVRerankerMTEBConfig, device: str = "cuda"):
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

        rwkv = RWKV_x070(self.args)
        reranker = RankingHead(self.args.head_size_a, self.args.n_embd)
        state = torch.load(cfg.reranker_path, map_location=self.device, weights_only=True)
        reranker.load_state_dict(state, strict=False)
        self.model = RWKVReranker(rwkv, reranker).to(self.device)

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

        return batch_ids

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
        if self.task2tag[task_name] != "[RETR]": # for non-retrieval tasks, add instruction
            instruct = self.task2instruct[task_name]
            if instruct and "Instruct:" not in sentences[0]:
                # avoid double instruction
                sentences = [instruct.format(query=s) for s in sentences]

        print(task_name, prompt_type, sentences[0][:200])
        for i in tqdm(range(0, len(sentences), batch_size), desc=task_name):
            batch_texts = sentences[i : i + batch_size]
            batch_ids = self._build_batch(batch_texts)
            with torch.inference_mode():
                batch_states = self.model(batch_ids, state_reduction="last") # [B, H, S, S]
            if i == 0:
                _, H, S, S = batch_states.shape
                embeddings = np.zeros((len(sentences), H, S, S), dtype=np.float32)
            embeddings[i:i+len(batch_texts), :, :, :] = batch_states.cpu().numpy()
        return embeddings
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> torch.Tensor:
        """Compute similarity scores between query and corpus states.
        """
        A, B = a.shape[0], b.shape[0]
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, device=self.device)

        pairwise_scores = torch.zeros((A, B), device=self.device)
        for i in range(A):
            query_state = a[i : i + 1].expand(B, -1, -1, -1)  # [B, H, S, S]
            scores = self.model.score_pairs(query_state, b)  # [B, 1]
            pairwise_scores[i] = scores.squeeze(-1)
        print("debug pairwise_scores:", pairwise_scores[0][:5])
        return pairwise_scores.cpu()