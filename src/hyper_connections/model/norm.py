from abc import ABC, abstractmethod
import torch


class Norm(torch.nn.Module, ABC):
    def __init__(self, dim: int, eps: float = 1e-5, scaled: bool = True, biased: bool = False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scaled = scaled
        self.biased = biased

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class LayerNorm(Norm):
    def __init__(self, dim: int, eps: float = 1e-5, scaled: bool = True, biased: bool = False):
        super().__init__(dim, eps, scaled, biased)
        self.layer_norm = torch.nn.LayerNorm(dim, elementwise_affine=scaled, bias=biased, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)


class RMSNorm(Norm):
    def __init__(self, dim: int, eps: float = 1e-5, scaled: bool = True, biased: bool = False):
        super().__init__(dim, eps, scaled, biased)
        if scaled:
            self.scale = torch.nn.Parameter(torch.ones(dim))
        if biased:
            self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x / (self.dim**0.5) + self.eps
        x_normed = x / rms_x
        if self.scaled:
            x_normed = x_normed * self.scale
        if self.biased:
            x_normed = x_normed + self.bias
        return x_normed


NORMS: dict[str, type[Norm]] = {
    "layer_norm": LayerNorm,
    "rms_norm": RMSNorm,
}
