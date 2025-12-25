########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, os, re, copy, glob, gzip, random, math
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from typing import Dict, List, Sequence, Any
from collections import defaultdict
from pathlib import Path
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 65534 # shift from -200 to 65535
DEFAULT_IMAGE_TOKEN = "<image>"
STOP_TOKEN_INDEX = 261
DEFAULT_STOP_TOKEN = "\n\n"
EOS_INDEX = 65535
EOS_TOKEN = "<eos>"


def pad_2_max_len(input_ids, max_len, pad_token_id):
    padding_len = max_len - len(input_ids)
    if padding_len <= 0:
        return input_ids
    # left pad with pad_token_id
    return [pad_token_id] * padding_len + input_ids


def process_textonly(
    query_text,
    positive_text,
    tokenizer,
    ctx_len,
    eos_chunk_size=None,
    do_pad_to_max_length=True,
    pad_token_id=0,
):
    """Tokenize query and positive texts and insert multiple EOS tokens.

    Args:
        query_text: Query text string.
        positive_text: Positive text string.
        tokenizer: Tokenizer with an ``encode`` method.
        ctx_len: Maximum context length of the model.
        eos_chunk_size: Insert an EOS token every ``chunk_size`` tokens.
            If ``None`` or ``<=0`` only a single EOS is appended at the end.
        do_pad_to_max_length: If ``True`` pad to ``ctx_len`` using ``pad_token_id``.
        pad_token_id: Token id used for padding.
    """

    chunk_size = eos_chunk_size if eos_chunk_size and eos_chunk_size > 0 else ctx_len

    def _tokenize_and_chunk(text: str):
        raw_tokens = list(tokenizer.encode(text))

        # avoid zero chunk sizes and compute how many EOS tokens to keep per sequence
        n_eos = ctx_len // chunk_size

        # limit token length so that we never exceed the context window once EOS tokens are added
        max_allowed_len = max(0, ctx_len - n_eos)
        t_len = min(len(raw_tokens), max_allowed_len)
        tokens = raw_tokens[-t_len:] if t_len > 0 else []

        def _normal_ctx_process(ids: List[int], n_eos: int):
            """Split ``ids`` evenly into ``n_eos`` chunks inserting EOS after each."""

            t_len = len(ids)
            input_ids: List[int] = []
            eos_mask: List[int] = []
            real_chunk_size = max(1, math.ceil(t_len / n_eos))
            for chunk_idx in range(n_eos):
                start = chunk_idx * real_chunk_size
                end = min(start + real_chunk_size, t_len)
                chunk = ids[start:end]
                input_ids.extend(chunk)
                input_ids.append(EOS_INDEX)
                eos_mask.append(1)
            return input_ids, eos_mask

        def _short_ctx_process(ids: List[int], n_eos: int):
            """Repeat ``ids`` so that every EOS has some preceding tokens."""

            input_ids: List[int] = []
            eos_mask: List[int] = []
            for _ in range(n_eos):
                input_ids.extend(ids)
                input_ids.append(EOS_INDEX)
                eos_mask.append(1)
            return input_ids, eos_mask

        t_len = len(tokens)
        if t_len <= max_allowed_len and t_len >= chunk_size:
            input_ids, eos_mask = _normal_ctx_process(tokens, n_eos)
        elif t_len < chunk_size:
            input_ids, eos_mask = _short_ctx_process(tokens, n_eos)
        else:
            raise NotImplementedError("Input too long, please check your data.")

        if do_pad_to_max_length and len(input_ids) < ctx_len:
            input_ids = pad_2_max_len(input_ids, ctx_len, pad_token_id)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(eos_mask, dtype=torch.bool)

    query_text_ids, query_eos_mask = _tokenize_and_chunk(query_text)
    positive_text_ids, positive_eos_mask = _tokenize_and_chunk(positive_text)

    return dict(
        query_ids=query_text_ids,
        positive_ids=positive_text_ids,
        query_eos_mask=query_eos_mask,
        positive_eos_mask=positive_eos_mask,
    )


class TextOnlyEmbedDatasetShard(Dataset):
    """Dataset that lazily loads jsonl.gz shards from a directory.

    Each process only keeps one shard in memory at a time. The shards are
    distributed across GPUs based on their index so that GPU ``i`` only reads
    files with ``index % world_size == i``. When all samples from the current
    shard are consumed the next shard for that GPU is loaded and the previous
    one is released to save memory.
    """

    def __init__(self, args):
        self.args = args
        self.vocab_size = args.vocab_size
        self.tokenizer = args.tokenizer

        # discover shards
        data_dir = Path(args.data_dir)
        self.shard_files = sorted(glob.glob(str(data_dir / "*.jsonl.gz")))
        if len(self.shard_files) == 0:
            raise ValueError(f"no jsonl.gz shards found in {data_dir}")

        self.current_shard_idx = -1  # so that first call to _load_next_shard loads shard 0
        self.list_data_dict: List[Dict[str, Any]] = []
        self.sample_ptr = 0

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    # ------------------------------------------------------------------
    def _load_next_shard(self, global_rank, world_size):
        """Load the next shard assigned to this rank into memory."""
        rank_shard_files = [f for i, f in enumerate(self.shard_files) if i % world_size == global_rank]
        self.current_shard_idx = (self.current_shard_idx + 1) % len(rank_shard_files)
        self.current_shard_path = rank_shard_files[self.current_shard_idx]

        # release previous data
        self.list_data_dict = []
        self.sample_ptr = 0

        with gzip.open(self.current_shard_path, "rt", encoding="utf-8") as f:
            for line in f:
                self.list_data_dict.append(json.loads(line))

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        # Distributed setup
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if self.sample_ptr >= len(self.list_data_dict):
            self._load_next_shard(global_rank, world_size)

        sample = self.list_data_dict[self.sample_ptr]
        self.sample_ptr += 1

        query_text = sample['instruction'] + sample['query_text']
        positive_text = sample['positive_text']

        data_dict = process_textonly(
            query_text,
            positive_text,
            tokenizer=self.tokenizer,
            ctx_len=self.args.ctx_len,
            eos_chunk_size=getattr(self.args, "eos_chunk_size", None),
            do_pad_to_max_length=True,
            pad_token_id=0,
        )

        data_dict['sample_id'] = str(sample['sample_id']) if 'sample_id' in sample else str(sample['id'])
        return data_dict
