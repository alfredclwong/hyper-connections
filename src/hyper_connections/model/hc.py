import einops
import huggingface_hub as hf
import torch
from typing import Callable

from hyper_connections.model.act import SwiGLU, ReLU
from hyper_connections.model.transformer import MLP, Attention
from hyper_connections.model.gpt import GPTConfig


class HCNet(torch.nn.Module):
    def __init__(self, dim, blocks, norm_gen: Callable[[], torch.nn.Module], pre_norm=True, expansion_rate=2, dynamic=False):
        super().__init__()
        self.n = expansion_rate
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self.norms = torch.nn.ModuleList([norm_gen() for _ in range(len(blocks))])
        self.pre_norm = pre_norm
        self.dynamic = dynamic

        self.A_m = torch.nn.Parameter(
            torch.stack(
                [torch.eye(self.n)[:, [i % self.n]] for i in range(self.num_blocks)]
            )
        )
        self.A_r = torch.nn.Parameter(
            torch.stack(
                [torch.eye(self.n) for _ in range(self.num_blocks)]
            )
        )
        self.B = torch.nn.Parameter(
            torch.stack(
                [torch.ones(1, self.n) for _ in range(self.num_blocks)]
            )
        )

        if self.dynamic:
            self.W_m = torch.nn.Parameter(
                torch.stack(
                    [torch.zeros(dim, 1) for _ in range(self.num_blocks)]
                )
            )
            self.W_r = torch.nn.Parameter(
                torch.stack(
                    [torch.zeros(dim, self.n) for _ in range(self.num_blocks)]
                )
            )
            self.W_b = torch.nn.Parameter(
                torch.stack(
                    [torch.zeros(dim, self.n) for _ in range(self.num_blocks)]
                )
            )
            self.s_a = torch.nn.Parameter(
                torch.stack(
                    [torch.ones(1, 1) * 0.01 for _ in range(self.num_blocks)]
                )
            )
            self.s_b = torch.nn.Parameter(
                torch.stack(
                    [torch.ones(1, 1) * 0.01 for _ in range(self.num_blocks)]
                )
            )
            self.dynamic_norms = torch.nn.ModuleList(
                torch.nn.LayerNorm(dim, bias=False) for _ in range(self.num_blocks)
            )

    def forward(self, x):
        H = x.unsqueeze(2).repeat(1, 1, self.n, 1)  # (B, T, n, C)
        for i, block in enumerate(self.blocks):
            h, H = self.width_connections(H, i)
            if self.pre_norm:
                h = self.norms[i](h)
            h = block(h)
            h = self.depth_connections(h, i)
            H = H + h
            if not self.pre_norm:
                H = self.norms[i](H)
        H = H.sum(dim=2)  # (B, T, C)
        return H

    def width_connections(self, H, block_idx):
        # TODO combine A_m and A_r into a single matrix to remove cat op
        A_m = self.A_m[block_idx]
        A_r = self.A_r[block_idx]
        if self.dynamic:
            H_norm = self.dynamic_norms[block_idx](H)
            A_m = A_m + self.s_a [block_idx] * torch.nn.functional.tanh(
                einops.einsum(H_norm, self.W_m[block_idx], "B T n C, C o -> B T n o")
            )
            A_r = A_r + self.s_a[block_idx] * torch.nn.functional.tanh(
                einops.einsum(H_norm, self.W_r[block_idx], "B T n C, C m -> B T n m")
            )
        WC = torch.cat([A_m, A_r], dim=-1)  # (n, n + 1)
        H = H.transpose(-2, -1) @ WC
        h = H[..., 0]  # (B, T, C)
        H = H[..., 1:]  # (B, T, C, n)
        H = H.transpose(-2, -1)  # (B, T, n, C)
        return h, H

    def depth_connections(self, h_BTC, block_idx):
        B = self.B[block_idx][0]
        if self.dynamic:
            h_norm = self.dynamic_norms[block_idx](h_BTC)
            B = B + self.s_b[block_idx] * torch.nn.functional.tanh(
                einops.einsum(h_norm, self.W_b[block_idx], "B T C, C n -> B T n")
            )
        H_BTnC = h_BTC.unsqueeze(2) * B.unsqueeze(-1)
        return H_BTnC


class HCTransformer(HCNet):
    def __init__(self, D, H, L, norm_gen, pre_norm: bool = True, R: int | None = None, expansion_rate: int = 2, dynamic: bool = False):
        blocks = torch.nn.ModuleList()
        for _ in range(L):
            blocks.append(Attention(D, H, R=R))
            blocks.append(MLP([D, D * 8 // 3, D], SwiGLU()))
            # blocks.append(MLP([D, D * 4, D], ReLU()))
        super().__init__(D, blocks, norm_gen=norm_gen, pre_norm=pre_norm, expansion_rate=expansion_rate, dynamic=dynamic)


class HCGPT(torch.nn.Module):
    def __init__(self, config: GPTConfig, expansion_rate=2, dynamic=False):
        super().__init__()
        self.token_emb = torch.nn.Embedding(config.vocab_size, config.dim, padding_idx=config.padding_idx)
        self.transformer = HCTransformer(
            config.dim,
            config.num_heads,
            config.num_layers,
            norm_gen=lambda: torch.nn.LayerNorm(config.dim, bias=False),
            pre_norm=True,
            R=config.base,
            expansion_rate=expansion_rate,
            dynamic=dynamic,
        )
        self.ln_f = torch.nn.LayerNorm(config.dim, bias=False)
        self.head = torch.nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, x, y=None):
        x = self.token_emb(x)
        x = self.transformer(x)
        x = self.ln_f(x)

        if y is None:  # inference
            logits = self.head(x[:, [-1], :])
            loss = None
        else:
            logits = self.head(x)
            log_probs = logits.log_softmax(dim=-1)
            loss = -log_probs.gather(-1, y.unsqueeze(-1)).mean()

        return logits, loss


class HubHCGPT(HCGPT, hf.PyTorchModelHubMixin):
    pass
