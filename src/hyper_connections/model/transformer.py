# Implement a transformer as a residual network to allow for HC

# B: batch size
# T: sequence length
# D: model dimension
# H: number of attention heads
# K: head dimension
# L: number of layers
# R: base for rotary embeddings
# N: expansion rate

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

    def forward(self, x_BTD: torch.Tensor) -> torch.Tensor:
        # x could be any (..., D) shape but for transformer we expect (B, T, D)
        for block, norm in zip(self.blocks, self.norms):
            if self.pre_norm:
                x_BTD = x_BTD + block(norm(x_BTD))
            else:
                x_BTD = norm(x_BTD + block(x_BTD))
        return x_BTD


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
            blocks.append(MLP([D, D * 8 // 3, D], SwiGLU()))
            # blocks.append(MLP([D, D * 4, D], ReLU()))
        super().__init__(blocks, norm_gen, pre_norm)


class Attention(torch.nn.Module):
    def __init__(self, D: int, H: int, R: int | None = None, flash: bool = True):
        super().__init__()
        self.H = H
        self.D = D
        self.K = D // H
        self.flash = flash

        self.rotary = None if R is None else Rotary(self.K, base=R)
        self.qkv = torch.nn.Linear(D, 3 * self.H * self.K, bias=False)
        self.out = torch.nn.Linear(self.D, D, bias=False)

    def forward(self, x_BTD: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, D = x_BTD.shape

        qkv_BTHK = self.qkv(x_BTD).view(B, T, 3, self.H, self.K)
        q_BTHK, k_BTHK, v_BTHK = qkv_BTHK.unbind(dim=2)
        if self.rotary is not None:
            cos, sin = self.rotary(q_BTHK, seq_dim=1)
            q_BTHK, k_BTHK = apply_rotary_pos_emb(q_BTHK, k_BTHK, cos, sin)

        if self.flash and torch.cuda.is_available() and torch.__version__ >= "2.0.0":
            attn_output_BTHK = torch.nn.functional.scaled_dot_product_attention(
                q_BTHK, k_BTHK, v_BTHK, is_causal=causal
            )
        else:
            attn_scores_BTHH = einops.einsum(
                q_BTHK, k_BTHK, "B T H K, B M H K -> B T H M"
            ) / (self.K**0.5)
            if causal:
                mask_TT = torch.tril(torch.ones(T, T, device=x_BTD.device)).bool()
                attn_scores_BTHH = attn_scores_BTHH.masked_fill(
                    ~mask_TT[None, :, None, :], float("-inf")
                )
            attn_probs_BTHH = torch.nn.functional.softmax(attn_scores_BTHH, dim=-1)
            attn_output_BTHK = einops.einsum(
                attn_probs_BTHH, v_BTHK, "B T H M, B M H K -> B T H K"
            )
        attn_output_BTD = attn_output_BTHK.reshape(B, T, D)
        output_BTD = self.out(attn_output_BTD)
        return output_BTD

class MLP(torch.nn.Sequential):
    def __init__(self, dims: list[int], act: Activation):
        super().__init__()
        for d0, d1 in zip(dims[:-2], dims[1:-1]):
            d1 *= act.pre_expansion_factor
            self.append(torch.nn.Linear(d0, d1, bias=False))
            self.append(act)
        self.append(torch.nn.Linear(*dims[-2:], bias=False))
