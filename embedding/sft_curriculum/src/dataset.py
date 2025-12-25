########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, os, re, copy, glob, gzip, random, math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from typing import Dict, List, Any
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
    negative_texts: Sequence[str] | None = None,
):
    """Tokenize query/positive texts and optional negatives inserting EOS tokens.

    Args:
        query_text: Query text string.
        positive_text: Positive text string.
        tokenizer: Tokenizer with an ``encode`` method.
        ctx_len: Maximum context length of the model.
        eos_chunk_size: Insert an EOS token every ``chunk_size`` tokens. If
            ``None`` or ``<=0`` only a single EOS is appended at the end.
        do_pad_to_max_length: If ``True`` pad to ``ctx_len`` using ``pad_token_id``.
        pad_token_id: Token id used for padding.
        negative_texts: Optional iterable of negative text strings.
    """

    chunk_size = eos_chunk_size if eos_chunk_size and eos_chunk_size > 0 else ctx_len

    def _tokenize_and_chunk(text: str):
        max_n_eos = math.ceil(ctx_len / chunk_size)
        if not text: # handle empty text
            input_ids = [EOS_INDEX] * max_n_eos
            eos_mask = [0] * max_n_eos
            if do_pad_to_max_length and len(input_ids) < ctx_len:
                input_ids = pad_2_max_len(input_ids, ctx_len, pad_token_id)
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(eos_mask, dtype=torch.bool)
                
        tokens = list(tokenizer.encode(text))
        # step 1: truncate first, avoid exceed ctx_len
        max_allowed_len = ctx_len - max_n_eos
        t_len = min(len(tokens), max_allowed_len)
        tokens = tokens[-t_len:] if t_len > 0 else []
        n_eos = max(1, math.ceil(t_len / chunk_size))
        # step 2: insert unused EOS at beginning
        input_ids: List[int] = []
        eos_mask: List[int] = []
        for _ in range(max_n_eos - n_eos):
            input_ids.append(EOS_INDEX)
            eos_mask.append(0)  # not used
        # step 3: append EOS at the end of each chunk
        # evenly distribute EOS tokens
        real_chunk_size = math.ceil(t_len / n_eos)
        chunks = [tokens[i : i + real_chunk_size] for i in range(0, t_len, real_chunk_size)]
        for chunk in chunks:
            input_ids.extend(chunk)
            input_ids.append(EOS_INDEX)
            eos_mask.append(1)
        # step 4: pad to ctx_len
        if do_pad_to_max_length and len(input_ids) < ctx_len:
            input_ids = pad_2_max_len(input_ids, ctx_len, pad_token_id)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(eos_mask, dtype=torch.bool)

    query_text_ids, query_eos_mask = _tokenize_and_chunk(query_text)
    positive_text_ids, positive_eos_mask = _tokenize_and_chunk(positive_text)

    data_dict = dict(
        query_ids=query_text_ids,
        positive_ids=positive_text_ids,
        query_eos_mask=query_eos_mask,
        positive_eos_mask=positive_eos_mask,
    )

    if negative_texts:
        for neg_idx, neg_text in enumerate(negative_texts, start=1):
            negative_ids, negative_eos_mask = _tokenize_and_chunk(neg_text)
            data_dict[f"negative{neg_idx}_ids"] = negative_ids
            data_dict[f"negative{neg_idx}_eos_mask"] = negative_eos_mask

    return data_dict


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
        self.num_negatives = args.num_negatives

        # discover shards
        data_dir = Path(args.data_dir)
        self.shard_files = sorted(glob.glob(str(data_dir / "*.jsonl.gz")))
        if len(self.shard_files) == 0:
            raise ValueError(f"no jsonl.gz shards found in {data_dir}")

        self.current_shard_idx = -1  # so that first call to _load_next_shard loads shard 0
        self.list_data_dict: List[Dict[str, Any]] = []
        self.sample_ptr = 0
        # multi-task head
        self.task_keys = ["[CLS]", "[STS]", "[RETR]"]
        self.task_weights = [1.0, 1.0, 1.0]
        self.task2id = {k: i for i, k in enumerate(self.task_keys)}
        self.task2weight = {k: w for k, w in zip(self.task_keys, self.task_weights)}

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
        source_list = []

        with gzip.open(self.current_shard_path, "rt", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                self.list_data_dict.append(j)
                source_list.append(j.get("source"))

        self.source_keys = sorted(list(set(source_list)))
        self.source2id = {k: i for i, k in enumerate(self.source_keys)}

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        # Distributed setup
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if self.sample_ptr >= len(self.list_data_dict):
            self._load_next_shard(global_rank, world_size)

        sample = self.list_data_dict[self.sample_ptr]
        self.sample_ptr += 1

        query_text = sample["query"]
        if 'instruct' in sample:
            query_text = sample['instruct'].format(query=query_text)

        pos_list = sample.get("pos")
        if isinstance(pos_list, Sequence) and not isinstance(pos_list, (str, bytes)) and len(pos_list) > 0:
            positive_text = random.choice(pos_list)
        elif isinstance(pos_list, str):
            positive_text = pos_list
        else:
            raise ValueError("Sample missing positive texts: expecting non-empty 'pos'.")

        neg_list = sample.get("neg", [])
        negative_texts: List[str] = []
        if self.num_negatives > 0 and isinstance(neg_list, Sequence) and not isinstance(neg_list, (str, bytes)):
            if len(neg_list) >= self.num_negatives:
                negative_texts = neg_list[:self.num_negatives]
            elif len(neg_list) >= 0:
                negative_texts = list(neg_list)
                # select random negatives to fill up
                while len(negative_texts) < self.num_negatives:
                    random_idx = random.randrange(0, len(self.list_data_dict))
                    random_sample = self.list_data_dict[random_idx]
                    negative_texts.extend(random_sample.get("neg", []))
                negative_texts = negative_texts[: self.num_negatives]

        data_dict = process_textonly(
            query_text,
            positive_text,
            tokenizer=self.tokenizer,
            ctx_len=self.args.ctx_len,
            eos_chunk_size=self.args.eos_chunk_size,
            do_pad_to_max_length=True,
            pad_token_id=0,
            negative_texts=negative_texts,
        )

        sample_identifier = sample.get('sample_id', sample.get('id'))
        if sample_identifier is not None:
            data_dict['sample_id'] = str(sample_identifier)
        # add task info if any
        if 'task' in sample:
            data_dict['task'] = self.task2id[sample['task']]
            data_dict['weight'] = self.task2weight[sample['task']]
        if 'source' in sample:
            data_dict['source'] = self.source2id[sample['source']]
        return data_dict