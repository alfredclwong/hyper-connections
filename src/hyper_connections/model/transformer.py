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

from hyper_connections.model.act import Activation, SwiGLU, ReLU
from hyper_connections.model.rope import Rotary, apply_rotary_pos_emb


class ResNet(torch.nn.Module):
    """
    A residual network is a sequence of blocks with skip connections.
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
            if self.pre_norm:
                x_BND = x_BND + block(norm(x_BND))
            else:
                x_BND = norm(x_BND + block(x_BND))
        return x_BND


class Transformer(ResNet):
    def __init__(
        self,
        D: int,
        H: int,
        L: int,
        norm_gen: Callable[[], torch.nn.Module],
        pre_norm: bool = True,
        R: int | None = None,
    ):
        blocks = torch.nn.ModuleList()
        for _ in range(L):
            blocks.append(Attention(D, H, R))
            blocks.append(MLP([D, D // 3 * 8, D], SwiGLU()))
            # blocks.append(MLP([D, D * 4, D], ReLU()))
        super().__init__(blocks, norm_gen, pre_norm)


class Attention(torch.nn.Module):
    def __init__(self, D: int, H: int, R: int | None = None):
        super().__init__()
        self.H = H
        self.D = D
        self.K = D // H

        self.rotary = None if R is None else Rotary(self.K, base=R)
        self.qkv = torch.nn.Linear(D, 3 * self.H * self.K, bias=False)
        self.out = torch.nn.Linear(self.D, D, bias=False)

    def forward(self, x_BND: torch.Tensor, causal: bool = True, flash: bool = False) -> torch.Tensor:
        B, N, D = x_BND.shape

        qkv_BNHK = self.qkv(x_BND).view(B, N, 3, self.H, self.K)
        q_BNHK, k_BNHK, v_BNHK = qkv_BNHK.unbind(dim=2)

        if self.rotary is not None:
            cos, sin = self.rotary(q_BNHK, seq_dim=1)
            q_BNHK, k_BNHK = apply_rotary_pos_emb(q_BNHK, k_BNHK, cos, sin)

        if flash and torch.cuda.is_available() and torch.__version__ >= "2.0.0":
            attn_output_BNHK = torch.nn.functional.scaled_dot_product_attention(
                q_BNHK, k_BNHK, v_BNHK, is_causal=causal
            )
        else:
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
