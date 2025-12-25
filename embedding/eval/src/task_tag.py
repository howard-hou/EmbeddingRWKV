import json, time, random, os
import numpy as np
import dataclasses
from torch.nn import functional as F
from typing import List, Dict
from collections import defaultdict
from PIL import Image
from pathlib import Path
import logging


def get_default_tag(task_description: str) -> str:
    if not task_description:
        return ''
    return '{}'.format(task_description)


def get_task_tag_by_task(task) -> str:
    task_name = Path(task.metadata.name).stem # remove version suffix if any
    task_type = task.metadata.type
    if task_type == 'Classification':
        return get_default_tag("[CLS]")
    if task_type == 'Clustering':
        return get_default_tag("[CLS]")
    if task_type == 'Retrieval':
        return get_default_tag("[RETR]")
    if task_type == 'Reranking':
        return get_default_tag("[RETR]")
    if task_type == 'STS':
        return get_default_tag("[STS]")
    if task_type == 'BitextMining':
        return get_default_tag("[STS]")
    if task_type == 'PairClassification':
        return get_default_tag("[STS]")
    if task_type == 'MultilabelClassification':
        return get_default_tag("[CLS]")
    if task_type == 'Regression':
        return get_default_tag("[CLS]")
    if task_type == 'InstructionRetrieval':
        return get_default_tag("[RETR]")
    if task_type == 'Summarization':
        return get_default_tag("[STS]")
    logging.warning(f"Unknown task type {task_type}, use None tag.")
    return get_default_tag(None)

