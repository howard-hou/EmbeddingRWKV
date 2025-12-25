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
        tokens = list(tokenizer.encode(text))
        max_n_eos = math.ceil(ctx_len / chunk_size)
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
        #
        self.template = "Query: {}\nDocument: {}"
        self.instruction_template = "Instruct: {}\nDocument: {}\nQuery: {}"

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz
    
    def sample_easy_negatives(self, positive_text: str, num_easy_negatives: int) -> List[str]:
        """Sample easy negatives from the dataset."""
        easy_negatives = []
        while len(easy_negatives) < num_easy_negatives:
            random_idx = random.randrange(0, len(self.list_data_dict))
            random_sample = self.list_data_dict[random_idx]
            candidate_texts = set(random_sample["pos"] + random_sample.get("neg", []))
            negative_text = random.choice(list(candidate_texts))
            # avoid sampling the positive text
            if negative_text != positive_text:
                easy_negatives.append(negative_text)
        return easy_negatives[:num_easy_negatives]
    
    def sample_medium_negatives(self, positive_text: str, source: str, query_text: str, num_medium_negatives: int) -> List[str]:
        """Sample medium negatives from same source."""
        medium_negatives = []
        candidates = self.source2data[source]
        # avoid sampling from the same query
        candidates = [sample for sample in candidates if sample["query"] != query_text]

        while len(medium_negatives) < num_medium_negatives:
            random_idx = random.randrange(0, len(candidates))
            random_sample = candidates[random_idx]
            candidate_texts = set(random_sample["pos"] + random_sample.get("neg", []))
            negative_text = random.choice(list(candidate_texts))
            # avoid sampling the positive text
            if negative_text != positive_text:
                medium_negatives.append(negative_text)
        return medium_negatives[:num_medium_negatives]

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
        self.source2data = defaultdict(list)

        with gzip.open(self.current_shard_path, "rt", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                self.list_data_dict.append(j)
                source_list.append(j.get("source"))
                self.source2data[j.get("source")].append(j)

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

        pos_list = sample.get("pos")
        if isinstance(pos_list, Sequence) and not isinstance(pos_list, (str, bytes)) and len(pos_list) > 0:
            positive_text = random.choice(pos_list)
        elif isinstance(pos_list, str):
            positive_text = pos_list
        else:
            raise ValueError("Sample missing positive texts: expecting non-empty 'pos'.")
        if 'instruct' in sample:
            query_positive = self.instruction_template.format(sample['instruct'], positive_text, query_text)
        else:
            query_positive = self.template.format(query_text, positive_text)

        # prepare negatives
        num_easy_negatives = int(self.num_negatives.split(':')[0])
        num_medium_negatives = int(self.num_negatives.split(':')[1])
        num_hard_negatives = int(self.num_negatives.split(':')[2])
        num_negatives = num_easy_negatives + num_medium_negatives + num_hard_negatives
        negative_texts = []
        if num_easy_negatives > 0:
            easy_negatives = self.sample_easy_negatives(positive_text, num_easy_negatives)
            negative_texts.extend(easy_negatives)
        if num_medium_negatives > 0:
            medium_negatives = self.sample_medium_negatives(positive_text, sample["source"], query_text, num_medium_negatives)
            negative_texts.extend(medium_negatives)
        if num_hard_negatives > 0 and "neg" in sample:
            hard_negatives = sample["neg"][:num_hard_negatives]
            negative_texts.extend(hard_negatives)
        # some time there are not enough hard negatives
        if len(negative_texts) < num_negatives:
            extra_needed = num_negatives - len(negative_texts)
            extra_negatives = self.sample_easy_negatives(positive_text, extra_needed)
            negative_texts.extend(extra_negatives)
        negative_texts = negative_texts[:num_negatives]
        if 'instruct' in sample:
            query_negatives = [self.instruction_template.format(sample['instruct'], neg, query_text) for neg in negative_texts]
        else:
            query_negatives = [self.template.format(query_text, neg) for neg in negative_texts]

        data_dict = process_textonly(
            query_text,
            query_positive,
            tokenizer=self.tokenizer,
            ctx_len=self.args.ctx_len,
            eos_chunk_size=self.args.eos_chunk_size,
            do_pad_to_max_length=True,
            pad_token_id=0,
            negative_texts=query_negatives,
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
