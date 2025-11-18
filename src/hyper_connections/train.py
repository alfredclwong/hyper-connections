"""
Rental GPU checklist:
- hf auth login
- wandb login
- batch_size
- use_wandb = True
- save_pretrained push_to_hub=True
- flash = True
- run with &
"""

# %%
from dataclasses import dataclass

import torch
import wandb
from tqdm.auto import tqdm

from hyper_connections.eval import eval
from hyper_connections.model.gpt import GPTConfig, HubGPT, HubHCGPT
from hyper_connections.util import (
    estimate_max_memory_usage,
    get_device,
    get_num_params,
    get_tokenized_c4_val_dataset,
    get_tokenized_dolma_train_dataset,
    get_tokenizer,
)

# %%
device = get_device()
device


# %%
@dataclass(frozen=True)
class TrainConfig:
    n_tokens: int = int(5e7)
    max_seq_len: int = 1024
    batch_size: int = 1
    val_n_tokens: int = 1024 * 512
    val_batch_size: int = 1
    n_epoch: int = 1
    lr: float = 3e-4
    weight_decay: float = 0
    betas: tuple[float, float] = (0.9, 0.999)
    use_wandb: bool = False


train_cfg = TrainConfig()

# %%
val_dataset = get_tokenized_c4_val_dataset(
    tokenizer=get_tokenizer(),
    n_tokens=train_cfg.val_n_tokens,
    batch_size=train_cfg.val_batch_size,
    seq_len=train_cfg.max_seq_len,
)

# %%
model_cfg = GPTConfig(
    # vocab_size=tokenizer.vocab_size,
    vocab_size=50304,  # 128 * 393
    dim=1024,
    num_heads=16,
    num_layers=12,
    base=10_000,
    padding_idx=1,
    expansion_rate=2,
    dynamic=True,
)
# model = HubGPT(model_cfg)
model = HubHCGPT(model_cfg)
num_params_m = get_num_params(model) // 1_000_000
print(f"{num_params_m}M parameters.")
model.to(device)

# %%
mem_usage_mb = estimate_max_memory_usage(model, batch_size=4)
print(f"Estimated max memory usage: {mem_usage_mb:.2f} MB")

# %%
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg.lr,
    weight_decay=train_cfg.weight_decay,
    betas=train_cfg.betas,
)
n_checkpoint_tokens = train_cfg.n_epoch * train_cfg.n_tokens // 10
n_eval_tokens = train_cfg.n_epoch * train_cfg.n_tokens // 25
next_checkpoint_tokens = n_checkpoint_tokens
next_eval_tokens = n_eval_tokens
if train_cfg.use_wandb:
    wandb.init(
        project="hyper-connections", config=model_cfg.__dict__ | train_cfg.__dict__
    )

n_tokens_seen = 0
val_loss = float("inf")
val_ppl = float("inf")
model.train()
for i in range(train_cfg.n_epoch):
    train_dataset = get_tokenized_dolma_train_dataset(
        tokenizer=get_tokenizer(),
        n_tokens=train_cfg.n_tokens,
        batch_size=train_cfg.batch_size,
        seq_len=train_cfg.max_seq_len,
    )
    pbar = tqdm(train_dataset)
    for input_ids in pbar:
        numel = input_ids.numel()
        n_tokens_seen += numel

        input_ids = input_ids.to(device)
        optimizer.zero_grad()
        logits, loss = model(input_ids[:, :-1], y=input_ids[:, 1:])
        loss.backward()
        optimizer.step()

        if n_tokens_seen >= next_eval_tokens:
            next_eval_tokens += n_eval_tokens
            val_loss = eval(model, val_dataset, device)
            if train_cfg.use_wandb:
                wandb.log(
                    {
                        "val_loss": val_loss,
                    },
                    step=n_tokens_seen,
                )

        pbar.set_description(f"Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

        if train_cfg.use_wandb:
            wandb.log(
                {
                    "loss": loss.item(),
                },
                step=n_tokens_seen,
            )

        if n_tokens_seen >= next_checkpoint_tokens:
            next_checkpoint_tokens += n_checkpoint_tokens
            n_tokens_seen_m = n_tokens_seen // 1_000_000
            model.save_pretrained(
                f"awonga/{model.__class__.__name__}-{num_params_m}M-{n_tokens_seen_m}Mtok"
            )
model.save_pretrained(
    f"awonga/{model.__class__.__name__}-{num_params_m}M-final", push_to_hub=True
)

if train_cfg.use_wandb:
    wandb.finish()

# %%
