import json, time, random, os
import torch
import numpy as np
import dataclasses
from torch.nn import functional as F
from typing import List, Dict, Union
from collections import defaultdict
from PIL import Image
from io import BytesIO
from pytorch_lightning.utilities import rank_zero_info
# possible image configurations: (1:2), (2:1), (1:1), (1:3), (3:1)
POSSIBLE_RESOLUTIONS = [(448, 896), (896, 448), (896, 896), (448, 1344), (1344, 448)]

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

def largest_3n_plus_2_prime(x):
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    # Integer division to obtain an integer n such that 3n+2 < x
    n = x // 3  
    while True:
        num = 3 * n + 2
        if num < x and is_prime(num):
            return num
        n -= 1


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    id: str
    roles: List[str]
    conversations: List[Dict[str, str]]

    def append_message(self, role, message):
        d = {"from": role, "value": message}
        self.conversations.append(d)


def compress_parameter_names(parameter_names):
    compressed = defaultdict(set)
    for weight in parameter_names:
        parts = weight.split('.')
        # find the block number which is a number
        split_index = None
        for i, part in enumerate(parts):
            if part.isdigit():
                block = part
                split_index = i
                break
        if split_index is not None:
            block = parts[split_index]  # 提取block号
            rest = '.'.join(parts[split_index+1:])  # 剩余部分
            prefix = '.'.join(parts[:split_index]) # 
            compressed[(prefix, rest)].add(block)
        else:
            compressed[(weight, '')].add('')

    # 格式化输出，合并具有相同rest部分的block号
    output = []
    for (prefix, rest), blocks in compressed.items():
        if rest and blocks:
            blocks = sorted([int(b) for b in blocks])
            block_range = '{' + ','.join(map(str, blocks)) + '}'
            output.append(f'{prefix}.{block_range}.{rest}')
        else:
            output.append(prefix)
    return output


def parse_reranker_layer_idx(layer_idx: Union[str, List[int]]) -> List[int]:
    """Parse comma-separated reranker layer indices into a list of ints."""

    if isinstance(layer_idx, list):
        indices = layer_idx
    elif isinstance(layer_idx, str):
        cleaned = [part.strip() for part in layer_idx.split(',') if part.strip()]
        if not cleaned:
            raise ValueError("reranker_layer_idx must contain at least one layer index")
        try:
            indices = [int(part) for part in cleaned]
        except ValueError as exc:
            raise ValueError(f"Invalid reranker layer indices: {layer_idx}") from exc
    else:
        raise TypeError("reranker_layer_idx must be a string or a list of integers")

    return indices


def resolve_reranker_layer_indices(layer_indices: List[int], total_layers: int) -> List[int]:
    resolved: List[int] = []
    for idx in layer_indices:
        if idx < 0:
            idx = total_layers + idx
        if idx < 0 or idx >= total_layers:
            raise ValueError(
                f"Layer index {idx} is out of bounds for model with {total_layers} layers"
            )
        resolved.append(idx)
    return resolved


def load_reranker_blocks(raw_state, reranker_layer_idx: List[int]):
    """Remap ``raw_state`` to the reranker blocks defined by ``reranker_layer_idx``."""

    raw_block_nums = sorted([
        int(k.split('.')[1])
        for k in raw_state.keys()
        if k.startswith('blocks.')
    ])

    raw_L = max(raw_block_nums) + 1
    selected_blocks = resolve_reranker_layer_indices(reranker_layer_idx, raw_L)

    rank_zero_info(
        f"[ReRanker Loader] raw model layers={raw_L}, reranker layers={len(selected_blocks)}, "
        f"using raw blocks={selected_blocks}"
    )

    remapped_state = {}

    for idx_new, idx_raw in enumerate(selected_blocks):
        prefix_raw = f'blocks.{idx_raw}.'
        prefix_new = f'blocks.{idx_new}.'

        for k, v in raw_state.items():
            if k.startswith(prefix_raw):
                new_k = k.replace(prefix_raw, prefix_new)
                remapped_state[new_k] = v

    if 'emb.weight' in raw_state:
        remapped_state['emb.weight'] = raw_state['emb.weight'][-1:].clone()

    for k in ['ln_out.weight', 'ln_out.bias']:
        if k in raw_state:
            remapped_state[k] = raw_state[k]

    return remapped_state


