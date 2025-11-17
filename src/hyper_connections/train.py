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
import torch
from tqdm.auto import tqdm
import wandb
import huggingface_hub as hf

from transformers import OlmoForCausalLM

from hyper_connections.model.gpt import GPTConfig, HubGPT
from hyper_connections.model.hc import HubHCGPT
from hyper_connections.util import get_device, get_num_params, get_tokenizer, get_tokenized_dolma_train_dataset, get_tokenized_c4_val_dataset, get_olmo, estimate_max_memory_usage
from hyper_connections.eval import eval

# %%
device = get_device()
device

# # %%
# model = get_olmo(device)
# x = torch.randint(0, model.config.vocab_size, (1, 16), device=device)
# model.forward(x, labels=x)

# # %%
# token = input("Enter your Hugging Face token: ")
# hf.login(token=token)

# %%
# n_tokens = int(5e9)  # 5B tokens
n_tokens = int(5e7)
max_seq_len = 1024
batch_size = 1
val_dataset = get_tokenized_c4_val_dataset(
    tokenizer=get_tokenizer(),
    n_tokens=1024 * 512,
    batch_size=batch_size,
    seq_len=max_seq_len,
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
)
# model = HubGPT(model_cfg)
model = HubHCGPT(model_cfg, dynamic=False)
num_params_m = get_num_params(model) // 1_000_000
print(f"{num_params_m}M parameters.")
model.to(device)

# %%
mem_usage_mb = estimate_max_memory_usage(model, batch_size=4)
print(f"Estimated max memory usage: {mem_usage_mb:.2f} MB")

# %%
use_wandb = False
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
n_epoch = 1
n_checkpoint_tokens = n_epoch * n_tokens // 10
n_eval_tokens = n_epoch * n_tokens // 25
next_checkpoint_tokens = n_checkpoint_tokens
next_eval_tokens = n_eval_tokens
if use_wandb:
    wandb.init(project="hyper-connections")

n_tokens_seen = 0
val_loss = float("inf")
val_ppl = float("inf")
model.train()
for i in range(n_epoch):
    train_dataset = get_tokenized_dolma_train_dataset(
        tokenizer=get_tokenizer(),
        n_tokens=n_tokens,
        batch_size=batch_size,
        seq_len=max_seq_len,
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
            if use_wandb:
                wandb.log({
                    "val_loss": val_loss,
                }, step=n_tokens_seen)

        pbar.set_description(f"Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

        if use_wandb:
            wandb.log({
                "loss": loss.item(),
            }, step=n_tokens_seen)

        if n_tokens_seen >= next_checkpoint_tokens:
            next_checkpoint_tokens += n_checkpoint_tokens
            n_tokens_seen_m = n_tokens_seen // 1_000_000
            model.save_pretrained(f"awonga/{model.__class__.__name__}-{num_params_m}M-{n_tokens_seen_m}Mtok")
model.save_pretrained(f"awonga/{model.__class__.__name__}-{num_params_m}M-final")

if use_wandb:
    wandb.finish()

# %%
