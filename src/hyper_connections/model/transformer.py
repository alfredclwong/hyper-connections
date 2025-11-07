# Implement a transformer as a residual network to allow for HC

# B: batch size
# N: sequence length
# D: model dimension
# H: number of attention heads
# K: head dimension
# L: number of layers
# R: base for rotary embeddings

import torch
import einops
from typing import Callable

from hyper_connections.model.act import Activation, SwiGLU
from hyper_connections.model.rope import Rotary, apply_rotary_pos_emb


class ResNet(torch.nn.Module):
    """
    A residual network is a sequence of blocks with skip connections.

    Notation:
        - X: input, shape (B, N, D)
        - T_i: i-th block, maps (N, D) -> (N, D)

    Forward pass:
        for each block T_i:
            H = X
            if pre-norm:
                H = norm(H)
            H = T_i(H)

            X = X + H
            if not pre-norm:
                X = norm(X)
        return X
    """

    def __init__(
        self, blocks: torch.nn.ModuleList, norm_gen: Callable[[], torch.nn.Module], pre_norm: bool = True
    ):
        super().__init__()
        self.blocks = blocks
        self.norms = torch.nn.ModuleList([norm_gen() for _ in range(len(blocks))])
        self.pre_norm = pre_norm

    def forward(self, x_BND: torch.Tensor) -> torch.Tensor:
        for block, norm in zip(self.blocks, self.norms):
            h_BND = x_BND
            if self.pre_norm:
                h_BND = norm(h_BND)
            h_BND = block(h_BND)
            x_BND = x_BND + h_BND
            if not self.pre_norm:
                x_BND = norm(x_BND)
        return x_BND


class Transformer(ResNet):
    def __init__(
        self,
        D: int,
        H: int,
        L: int,
        R: int,
        norm: torch.nn.Module,
        pre_norm: bool = True,
    ):
        blocks = torch.nn.ModuleList()
        for _ in range(L):
            blocks.append(RotaryAttention(R, D, H))
            # blocks.append(MLP([D, D // 3 * 4, D], SwiGLU()))
            blocks.append(MLP([D, D * 4, D], SwiGLU()))
        super().__init__(blocks, norm, pre_norm)


class RotaryAttention(torch.nn.Module):
    def __init__(self, R: int, D: int, H: int):
        super().__init__()
        self.H = H
        self.D = D
        self.K = D // H

        self.rotary = Rotary(self.K, base=R)
        self.qkv = torch.nn.Linear(D, 3 * self.H * self.K, bias=False)
        self.out = torch.nn.Linear(self.D, D, bias=False)

    def forward(self, x_BND: torch.Tensor, causal: bool = True):
        B, N, D = x_BND.shape

        qkv_BNHK = self.qkv(x_BND).view(B, N, 3, self.H, self.K)
        q_BNHK, k_BNHK, v_BNHK = qkv_BNHK.unbind(dim=2)

        cos, sin = self.rotary(q_BNHK, seq_dim=1)
        q_BNHK, k_BNHK = apply_rotary_pos_emb(q_BNHK, k_BNHK, cos, sin)

        attn_scores_BNHH = einops.einsum(
            q_BNHK, k_BNHK, "B N H K, B M H K -> B N H M"
        ) / (self.K**0.5)

        if causal:
            mask_NM = torch.tril(torch.ones(N, N, device=x_BND.device)).bool()
            attn_scores_BNHH = attn_scores_BNHH.masked_fill(
                ~mask_NM[None, :, None, :], float("-inf")
            )

        attn_probs_BNHH = torch.nn.functional.softmax(attn_scores_BNHH, dim=-1)
        attn_output_BNHK = einops.einsum(
            attn_probs_BNHH, v_BNHK, "B N H M, B M H K -> B N H K"
        )
        attn_output_BND = attn_output_BNHK.reshape(B, N, D)
        output_BND = self.out(attn_output_BND)
        return output_BND


class MLP(torch.nn.Sequential):
    def __init__(self, dims: list[int], act: Activation):
        super().__init__()
        for d0, d1 in zip(dims[:-2], dims[1:-1]):
            d1 *= act.pre_expansion_factor
            self.append(torch.nn.Linear(d0, d1, bias=False))
            self.append(act)
        self.append(torch.nn.Linear(*dims[-2:], bias=False))
