import torch
import huggingface_hub as hf
from dataclasses import dataclass

from hyper_connections.model.transformer import Transformer


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    dim: int
    num_heads: int
    num_layers: int
    base: int | None = 10_000
    n_ctx: int | None = None
    padding_idx: int | None = None
    expansion_rate: int = 2
    dynamic: bool = False


class GPT(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_emb = torch.nn.Embedding(config.vocab_size, config.dim, padding_idx=config.padding_idx)
        if config.base is None:
            assert isinstance(config.n_ctx, int), "n_ctx must be provided if base is None"
            self.pos_emb = torch.nn.Embedding(config.n_ctx, config.dim)
        norm_gen = lambda: torch.nn.LayerNorm(config.dim, bias=False)
        self.transformer = Transformer(config.dim, config.num_heads, config.num_layers, norm_gen, pre_norm=True, R=config.base)
        self.ln_f = torch.nn.LayerNorm(config.dim, bias=False)
        self.head = torch.nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, x, y=None):
        x = self.token_emb(x)
        if hasattr(self, "pos_emb"):
            x = x + self.pos_emb.weight[None, : x.size(1), :]
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


class HubGPT(GPT, hf.PyTorchModelHubMixin):
    pass
