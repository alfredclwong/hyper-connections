from dataclasses import dataclass

import huggingface_hub as hf
import torch

from hyper_connections.model.norm import NORMS
from hyper_connections.model.transformer import Transformer
from hyper_connections.model.hc import HCTransformer


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    dim: int
    num_heads: int
    num_layers: int
    base: int | None = 10_000
    n_ctx: int | None = None
    flash: bool = True
    padding_idx: int | None = None
    expansion_rate: int = 2
    dynamic: bool = True
    norm: str = "layer_norm"


class HCGPT(torch.nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.token_emb = torch.nn.Embedding(
            cfg.vocab_size, cfg.dim, padding_idx=cfg.padding_idx
        )
        norm = NORMS[cfg.norm]
        self.transformer = HCTransformer(
            cfg.dim,
            cfg.num_heads,
            cfg.num_layers,
            norm=norm,
            pre_norm=True,
            R=cfg.base,
            expansion_rate=cfg.expansion_rate,
            dynamic=cfg.dynamic,
            flash=cfg.flash,
        )
        self.ln_f = norm(cfg.dim, scaled=True, biased=False)
        self.head = torch.nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

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


class GPT(torch.nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.token_emb = torch.nn.Embedding(
            cfg.vocab_size, cfg.dim, padding_idx=cfg.padding_idx
        )
        if cfg.base is None:
            assert isinstance(cfg.n_ctx, int), "n_ctx must be provided if base is None"
            self.pos_emb = torch.nn.Embedding(cfg.n_ctx, cfg.dim)
        norm = NORMS[cfg.norm]
        self.transformer = Transformer(
            cfg.dim,
            cfg.num_heads,
            cfg.num_layers,
            norm,
            pre_norm=True,
            R=cfg.base,
            flash=cfg.flash,
        )
        self.ln_f = norm(cfg.dim, scaled=True, biased=False)
        self.head = torch.nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

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


class HubHCGPT(HCGPT, hf.PyTorchModelHubMixin):
    pass


class HubGPT(GPT, hf.PyTorchModelHubMixin):
    pass
