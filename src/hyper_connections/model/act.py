import torch

from abc import abstractmethod


class Activation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def pre_expansion_factor(self) -> int:
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(x)

    @property
    def pre_expansion_factor(self) -> int:
        return 1


class GeLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

    @property
    def pre_expansion_factor(self) -> int:
        return 1


class SwiGLU(Activation):
    # note: input dim must be 2x output dim
    # we do this to avoid extra linear layer
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.silu(gate)

    @property
    def pre_expansion_factor(self):
        return 2
