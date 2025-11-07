import torch
import huggingface_hub as hf

from model.transformer import Transformer


class GPT(torch.nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, base, padding_idx=None):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.norm_gen = lambda: torch.nn.LayerNorm(dim, bias=False)
        self.transformer = Transformer(dim, num_heads, num_layers, base, self.norm_gen)
        self.ln_f = torch.nn.LayerNorm(dim, bias=False)
        self.head = torch.nn.Linear(dim, vocab_size, bias=False)

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


class HubGPT(GPT, hf.PyTorchModelHubMixin):
    pass
