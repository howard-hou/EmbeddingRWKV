from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class RankingHead(nn.Module):
    def __init__(self, head_size, n_embd):
        super().__init__()
        input_dim = head_size * head_size
        # ranking head
        self.ranking_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, n_embd, bias=False), 
            nn.ReLU(),
            nn.Linear(n_embd, 1, bias=False)
        )
        
    def forward(self, query_state: torch.Tensor, doc_state: torch.Tensor) -> torch.Tensor:
        # query_state: [B, H, S, S]
        # doc_state: [B, H, S, S]
        B, H, S, S = query_state.shape
        # cross features: [B, H, S, S], v_q @ v_d^T
        query_state = query_state.permute(0,2,1,3).reshape(B, S, H*S).contiguous()  # [B, S, H*S]
        doc_state = doc_state.permute(0,2,1,3).reshape(B, S, H*S).contiguous()      # [B, S, H*S]
        cross_features = torch.matmul(query_state, doc_state.mT)
        cross_features = cross_features.view(B, S * S) # [B, S*S]
        # fuse head dimension
        score_per_head = self.ranking_head(cross_features) # [B, 1]
        # fuse head scores
        score = score_per_head#.mean(dim=1)  # [B, 1]
        return score


def chunk_tokens(token_ids, chunk_size=512):
    batch_chunk_ids = []
    B, L = len(token_ids), len(token_ids[0])
    for s in range(0, L, chunk_size):
        chunks = []   
        for seq in token_ids:
            chunk = seq[s:s+chunk_size]
            chunks.append(chunk)
        batch_chunk_ids.append(chunks)
    return batch_chunk_ids


class RWKVReranker(nn.Module):
    """Lightning module that wraps the RWKV-based reranker for evaluation."""

    def __init__(self, rwkv, reranker) -> None:
        super().__init__()
        self.rwkv = rwkv
        self.reranker = reranker

    @torch.inference_mode()
    def encode_states_chunk(self, token_ids: List[List[int]], state_reduction) -> torch.Tensor:
        """Return the RWKV states for a batch of token sequences."""
        batch_size = len(token_ids)

        state = self.rwkv.generate_zero_state(batch_size)
        batch_chunk_ids = chunk_tokens(token_ids, chunk_size=512)
        chunk_states = []
        for chunk_ids in batch_chunk_ids:
            self.rwkv.forward_batch(chunk_ids, state)
            wkv_state = state[1].clone()  # [L, B, H, S, S]
            chunk_states.append(wkv_state)
        wkv_states = torch.stack(chunk_states, dim=0)  # [C, L, B, H, S, S]
        wkv_states = wkv_states.mean(dim=0)  # [L, B, H, S, S]
        out_state = self.reduce_state(wkv_states, state_reduction)  # [B, H, S, S]
        return out_state
    
    @torch.inference_mode()
    def encode_states(self, token_ids: List[List[int]], state_reduction) -> torch.Tensor:
        """Return the RWKV states for a batch of token sequences."""
        batch_size = len(token_ids)

        state = self.rwkv.generate_zero_state(batch_size)
        self.rwkv.forward_batch(token_ids, state)
        wkv_state = state[1].detach().clone()  # [L, B, H, S, S]
        out_state = self.reduce_state(wkv_state, state_reduction)  # [B, H, S, S]
        return out_state
    
    def reduce_state(self, state, state_reduction):
        # state: [Layers, B, H, S, S]
        if state_reduction == "mean":
            # mean pooling over heads and sequence dimensions
            return state.mean(dim=0) # [B, H, S, S]
        elif state_reduction == "concat":
            # concatenate heads and sequence dimensions
            Layers, B, H, S, S = state.shape
            return state.permute(1, 0, 2, 3, 4).reshape(B, Layers * H, S, S)  # [B, Layers*H, S, S]
        elif state_reduction == "last":
            # take the last layer
            return state[-1] # [B, H, S, S]
        else:
            raise ValueError(f"Unknown state reduction method: {self.args.state_reduction}")

    @torch.inference_mode()
    def score_pairs(self, query_state: torch.Tensor, doc_state: torch.Tensor) -> torch.Tensor:
        """Compute reranker logits for batches of query/document states."""
        return self.reranker(query_state, doc_state)

    @torch.inference_mode()
    def forward(self, batch_ids: List[List[int]], state_reduction) -> torch.Tensor:
        return self.encode_states(batch_ids, state_reduction)