"""Utilities for exporting the reranker sub-module from a VisualRWKV checkpoint.

This script extracts the weights that belong to the ``reranker`` module from a
VisualRWKV checkpoint and stores them in a standalone checkpoint that can be
loaded directly into :class:`torch.nn.Module.load_state_dict` of the reranker
component.  The script supports checkpoints produced by PyTorch Lightning as
well as plain ``state_dict`` checkpoints.

Example
-------
.. code-block:: bash

    python data_processing/export_reranker.py \
        --input in_visualrwkv.pth \
        --output out_reranker_only.pth

"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from typing import Iterable, MutableMapping, Optional

import torch

DEFAULT_STATE_DICT_KEYS: tuple[str, ...] = (
    "state_dict",
    "model_state_dict",
    "model",
    "module",
    "params",
)


def _resolve_state_dict(
    checkpoint: object, state_dict_key: Optional[str] = None
) -> MutableMapping[str, torch.Tensor]:
    """Return the underlying state dict from a checkpoint object."""

    if state_dict_key is not None:
        if not isinstance(checkpoint, MutableMapping) or state_dict_key not in checkpoint:
            raise KeyError(
                f"The provided state_dict_key '{state_dict_key}' is not present in the checkpoint."
            )
        state_dict = checkpoint[state_dict_key]
    elif isinstance(checkpoint, MutableMapping):
        state_dict = None
        for key in DEFAULT_STATE_DICT_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, MutableMapping):
                state_dict = value
                break
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, MutableMapping):
        raise TypeError(
            "Unable to locate a state_dict within the checkpoint. "
            "Pass --state-dict-key explicitly if the checkpoint structure is custom."
        )

    return state_dict


def _extract_module_state_dict(
    state_dict: MutableMapping[str, torch.Tensor], module_name: str
) -> "OrderedDict[str, torch.Tensor]":
    """Extract parameters that belong to ``module_name``."""

    extracted: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    module_token = module_name

    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue

        parts = key.split(".")
        try:
            module_index = parts.index(module_token)
        except ValueError:
            continue

        new_key_parts = parts[module_index + 1 :]
        if not new_key_parts:
            continue
        new_key = ".".join(new_key_parts)
        extracted[new_key] = value

    if not extracted:
        raise ValueError(
            f"No parameters for module '{module_name}' were found in the checkpoint."
        )

    metadata = getattr(state_dict, "_metadata", None)
    if isinstance(metadata, MutableMapping):
        extracted._metadata = OrderedDict()
        for meta_key, meta_value in metadata.items():
            if not isinstance(meta_key, str):
                continue
            parts = meta_key.split(".")
            try:
                module_index = parts.index(module_token)
            except ValueError:
                continue
            new_key = ".".join(parts[module_index + 1 :])
            extracted._metadata[new_key] = meta_value

    return extracted


DEFAULT_MODULE_NAME = "reranker"


def export_reranker_checkpoint(
    input_path: str,
    output_path: str,
    module_name: str = DEFAULT_MODULE_NAME,
    state_dict_key: Optional[str] = None,
) -> None:
    """Extract the reranker module from ``input_path`` and save it to ``output_path``."""

    checkpoint = torch.load(input_path, map_location="cpu")
    state_dict = _resolve_state_dict(checkpoint, state_dict_key)
    reranker_state = _extract_module_state_dict(state_dict, module_name)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    torch.save(reranker_state, output_path)


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the reranker weights from a VisualRWKV checkpoint.",
    )
    parser.add_argument(
        "input",
        help="Path to the VisualRWKV checkpoint that contains the reranker module.",
    )
    parser.add_argument(
        "output",
        help="Destination path for the exported reranker-only checkpoint.",
    )
    parser.add_argument(
        "--module-name",
        default=DEFAULT_MODULE_NAME,
        help=(
            "Name of the module inside the checkpoint to extract (default: reranker)."
        ),
    )
    parser.add_argument(
        "--state-dict-key",
        default=None,
        help=(
            "Optional key that directly references the state_dict inside the checkpoint. "
            "If omitted, the script searches common keys such as 'state_dict'."
        ),
    )
    return parser.parse_args(args=args)


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = parse_args(cli_args)
    export_reranker_checkpoint(
        input_path=args.input,
        output_path=args.output,
        module_name=args.module_name,
        state_dict_key=args.state_dict_key,
    )


if __name__ == "__main__":
    main()
