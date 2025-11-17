import einops
import torch

from hyper_connections.model.act import ReLU, SwiGLU
from hyper_connections.model.norm import Norm
from hyper_connections.model.transformer import MLP, Attention


class HCNet(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        blocks: torch.nn.ModuleList,
        norm: type[Norm],
        pre_norm: bool,
        expansion_rate: int,
        dynamic: bool,
    ):
        super().__init__()
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self.pre_norm = pre_norm
        self.dynamic = dynamic

        self.n = expansion_rate  # negative n = frac connections with m = -n
        frac_dim = dim if self.n > 0 else dim // -self.n

        # residual stream norms operate at full dim
        self.norms = torch.nn.ModuleList([norm(dim=dim) for _ in range(len(blocks))])

        if self.n > 0:
            self.A_m = torch.nn.Parameter(
                torch.stack(
                    [torch.eye(self.n)[:, [i % self.n]] for i in range(self.num_blocks)]
                )
            )  # (num_blocks, n, 1)
        else:
            self.A_m = torch.nn.Parameter(
                torch.stack([torch.eye(-self.n) for _ in range(self.num_blocks)])
            )  # (num_blocks, |n|, |n|)
        self.A_r = torch.nn.Parameter(
            torch.stack([torch.eye(abs(self.n)) for _ in range(self.num_blocks)])
        )
        self.B = torch.nn.Parameter(
            torch.stack([torch.ones(1, abs(self.n)) for _ in range(self.num_blocks)])
        )

        if self.dynamic:
            if self.n > 0:
                self.W_m = torch.nn.Parameter(
                    torch.stack([torch.zeros(frac_dim, 1) for _ in range(self.num_blocks)])
                )
            else:
                self.W_m = torch.nn.Parameter(
                    torch.stack(
                        [torch.zeros(frac_dim, -self.n) for _ in range(self.num_blocks)]
                    )
                )
            self.W_r = torch.nn.Parameter(
                torch.stack(
                    [torch.zeros(frac_dim, abs(self.n)) for _ in range(self.num_blocks)]
                )
            )
            self.W_b = torch.nn.Parameter(
                torch.stack(
                    [torch.zeros(frac_dim, abs(self.n)) for _ in range(self.num_blocks)]
                )
            )
            self.s_a = torch.nn.Parameter(
                torch.stack([torch.ones(1, 1) * 0.01 for _ in range(self.num_blocks)])
            )
            self.s_b = torch.nn.Parameter(
                torch.stack([torch.ones(1, 1) * 0.01 for _ in range(self.num_blocks)])
            )
            self.dynamic_norms = torch.nn.ModuleList(
                torch.nn.LayerNorm(frac_dim, bias=False) for _ in range(self.num_blocks)
            )

    def forward(self, x):
        # TODO make some contiguous calls
        if self.n > 0:
            return self._forward_hyper(x)
        else:
            return self._forward_frac(x)

    def _forward_hyper(self, x):
        assert self.n > 0
        H_BTND = x.unsqueeze(2).repeat(1, 1, self.n, 1)  # (B, T, n, D)
        for i, block in enumerate(self.blocks):
            h_BTD, H_BTND = self.width_connections(H_BTND, i)
            if self.pre_norm:
                h_BTD = self.norms[i](h_BTD)
            h_BTD = block(h_BTD)
            h_BDnD = self.depth_connections(h_BTD, i)
            H_BTND = H_BTND + h_BDnD
            if not self.pre_norm:
                H_BTND = self.norms[i](H_BTND)
        return H_BTND.sum(dim=2)  # (B, T, D)

    def _forward_frac(self, x):
        assert self.n < 0
        M = -self.n
        F = x.shape[-1] // M
        H_BTMF = x.view(*x.shape[:-1], M, F)
        for i, block in enumerate(self.blocks):
            h_BTD, H_BTMF = self.width_connections(H_BTMF, i)
            if self.pre_norm:
                h_BTD = self.norms[i](h_BTD)
            h_BTD = block(h_BTD)
            h_BTMF = h_BTD.view(*h_BTD.shape[:-1], M, F)
            h_BTMF = self.depth_connections(h_BTMF, i)
            H_BTMF = H_BTMF + h_BTMF
            if not self.pre_norm:
                H_BTMF = self.norms[i](H_BTMF)
        return H_BTMF.flatten(-2)  # (B, T, D)

    def width_connections(self, H, block_idx):
        # TODO combine A_m and A_r into a single matrix to remove cat op
        A_m = self.A_m[block_idx]
        A_r = self.A_r[block_idx]
        if self.dynamic:
            H_norm = self.dynamic_norms[block_idx](H)
            A_m = A_m + self.s_a[block_idx] * torch.nn.functional.tanh(
                einops.einsum(H_norm, self.W_m[block_idx], "... n D, D o -> ... n o")
            )
            A_r = A_r + self.s_a[block_idx] * torch.nn.functional.tanh(
                einops.einsum(H_norm, self.W_r[block_idx], "... n D, D m -> ... n m")
            )
        WC = torch.cat([A_m, A_r], dim=-1)  # (n, n + 1) or (-n, -2n)
        H = H.transpose(-2, -1) @ WC
        if self.n > 0:
            h = H[..., 0]  # (B, T, D)
        else:
            h = H[..., :-abs(self.n)].flatten(-2)  # (B, T, D)
        H = H[..., -abs(self.n):]  # (B, T, D, |n|)
        H = H.transpose(-2, -1)  # (B, T, |n|, D)
        return h, H

    def depth_connections(self, h, block_idx):
        # h is (B, T, D) or (B, T, M, F)
        B = self.B[block_idx][0]
        if self.dynamic:
            h_norm = self.dynamic_norms[block_idx](h)
            B = B + self.s_b[block_idx] * torch.nn.functional.tanh(
                einops.einsum(h_norm, self.W_b[block_idx], "... C, C n -> ... n")
            )  # C = D or F
        if self.n > 0:
            H = einops.einsum(h, B, "B T D, ... N -> B T N D")
        else:
            H = einops.einsum(h, B, "B T M F, ... M -> B T M F")
        return H


class HCTransformer(HCNet):
    def __init__(
        self,
        D: int,
        H: int,
        L: int,
        norm: type[Norm],
        pre_norm: bool,
        R: int | None,
        expansion_rate: int,
        dynamic: bool,
        flash: bool,
    ):
        blocks = torch.nn.ModuleList()
        for _ in range(L):
            blocks.append(Attention(D, H, R=R, flash=flash))
            blocks.append(MLP([D, D * 8 // 3, D], SwiGLU()))
            # blocks.append(MLP([D, D * 4, D], ReLU()))
        super().__init__(
            D,
            blocks,
            norm=norm,
            pre_norm=pre_norm,
            expansion_rate=expansion_rate,
            dynamic=dynamic,
        )
