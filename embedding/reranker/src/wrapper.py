import torch
import torch.nn as nn
import numpy as np
from typing import TYPE_CHECKING, Callable, Literal
from .dataset2 import EOS_INDEX
from sentence_transformers.model_card import SentenceTransformerModelCardData


class TokenizerWrapper:
    def __init__(self, tokenizer, template):
        self.tokenizer = tokenizer
        self.template = template
        self.max_len = 2048
        self.pad_token_id = 0

    def __call__(self, batch, padding=True, truncation=True, return_tensors="pt"):
        if len(batch[0]) == 3:
            batch = [(query, document) for query, document, _ in batch]
            
        batch_ids = []
        for query, document in batch:
            text = self.template.format(query=query, document=document)
            ids = list(self.tokenizer.encode(text))
            # add eos token
            if ids[-1] != EOS_INDEX:
                ids.append(EOS_INDEX)
            batch_ids.append(ids)

        # left padding
        if padding:
            batch_max_len = max(len(ids) for ids in batch_ids)
            padded_batch_ids = []
            for ids in batch_ids:
                pad_len = batch_max_len - len(ids)
                padded_ids = [self.pad_token_id] * pad_len + ids
                padded_batch_ids.append(padded_ids)
            batch_ids = padded_batch_ids
        
        # truncation
        if truncation:
            batch_ids = [ids[-self.max_len:] for ids in batch_ids]

        # return as tensor
        if return_tensors == "pt":
            batch_ids = torch.tensor(batch_ids, dtype=torch.long)
        return batch_ids


class CrossEncoderWrapper(nn.Module):
    def __init__(self, model, tokenizer, template):
        super(CrossEncoderWrapper, self).__init__()
        self.model = model
        self.tokenizer = TokenizerWrapper(tokenizer, template)
        self.model_card_data = SentenceTransformerModelCardData(
            model_name="RWKV-Reranker",
            language=["zh", "en"],
            license="MIT"
        )

    @torch.inference_mode()
    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]] | tuple[str, str] | list[str],
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        activation_fn: Callable | None = None,
        apply_softmax: bool | None = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
        """
        Performs predictions with the CrossEncoder on the given sentence pairs.

        Args:
            sentences (Union[List[Tuple[str, str]], Tuple[str, str]]): A list of sentence pairs [(Sent1, Sent2), (Sent3, Sent4)]
                or one sentence pair (Sent1, Sent2).
            batch_size (int, optional): Batch size for encoding. Defaults to 32.
            show_progress_bar (bool, optional): Output progress bar. Defaults to None.
            activation_fn (callable, optional): Activation function applied on the logits output of the CrossEncoder.
                If None, the ``model.activation_fn`` will be used, which defaults to :class:`torch.nn.Sigmoid` if num_labels=1, else
                :class:`torch.nn.Identity`. Defaults to None.
            convert_to_numpy (bool, optional): Convert the output to a numpy matrix. Defaults to True.
            apply_softmax (bool, optional): If set to True and `model.num_labels > 1`, applies softmax on the logits
                output such that for each sample, the scores of each class sum to 1. Defaults to False.
            convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, output
                a list of PyTorch tensors. Defaults to True.
            convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                Defaults to False.

        Returns:
            Union[List[torch.Tensor], np.ndarray, torch.Tensor]: Predictions for the passed sentence pairs.
            The return type depends on the ``convert_to_numpy`` and ``convert_to_tensor`` parameters.
            If ``convert_to_tensor`` is True, the output will be a :class:`torch.Tensor`.
            If ``convert_to_numpy`` is True, the output will be a :class:`numpy.ndarray`.
            Otherwise, the output will be a list of :class:`torch.Tensor` values.

        Examples:
            ::

                from sentence_transformers import CrossEncoder

                model = CrossEncoder("cross-encoder/stsb-roberta-base")
                sentences = [["I love cats", "Cats are amazing"], ["I prefer dogs", "Dogs are loyal"]]
                model.predict(sentences)
                # => array([0.6912767, 0.4303499], dtype=float32)
        """
        # Cast an individual pair to a list with length 1
        input_was_singular = False
        if sentences and isinstance(sentences, (list, tuple)) and isinstance(sentences[0], str):
            sentences = [sentences]
            input_was_singular = True

        pred_scores = []
        self.eval()
        for start_index in range(0, len(sentences), batch_size):
            batch = sentences[start_index : start_index + batch_size]
            batch_ids = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_ids = batch_ids.to(self.model.device)
            logits = self.model.predict(batch_ids)
            pred_scores.extend(logits)

        if convert_to_tensor:
            if len(pred_scores):
                pred_scores = torch.stack(pred_scores)
            else:
                pred_scores = torch.tensor([], device=self.model.device)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().float().numpy() for score in pred_scores])

        if input_was_singular:
            pred_scores = pred_scores[0]

        return pred_scores
