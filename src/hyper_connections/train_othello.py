# %%
import torch
from tqdm.auto import tqdm
import wandb
import huggingface_hub as hf
from datasets import load_dataset
from dataclasses import dataclass
import weave

from hyper_connections.model.hc import HubHCGPT
from hyper_connections.model.gpt import HubGPT, GPTConfig
from hyper_connections.util import get_device, get_num_params

# %%
device = get_device()
device

# %%
@dataclass(frozen=True)
class TrainConfig:
    n_epoch: int = 2
    n_train: int = 1_500_000
    n_val: int = 10_000
    lr: float = 1e-3
    weight_decay: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    use_wandb: bool = True
    val_steps: int | float = 0.1

train_cfg = TrainConfig()

# %%
train_dataset = load_dataset("awonga/othello-gpt-redux", data_dir="data", split=f"train[:{train_cfg.n_train}]")
train_dataset = train_dataset.with_format("torch")
val_dataset = load_dataset("awonga/othello-gpt-redux", data_dir="data", split=f"test[:{train_cfg.n_val}]")
val_dataset = val_dataset.with_format("torch")

# %%
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=512,
    shuffle=False,
    drop_last=True,
)

# %%
def input_ids_to_move_ids(input_ids: torch.Tensor, size: int):
    nw_center = size * (size // 2 - 1) + size // 2 - 1
    sw_center = nw_center + size
    move_ids = input_ids.clone()
    move_ids[move_ids >= nw_center] += 2
    move_ids[move_ids >= sw_center] += 2
    return move_ids

all_squares = input_ids_to_move_ids(torch.arange(32), 6).to(device)

@torch.no_grad()
def eval(model, val_loader):
    val_loss = 0.0
    val_prob_legal = 0.0
    val_top1_legal = 0.0
    n_batches = 0
    model.eval()
    for batch in tqdm(val_loader, desc="Evaluating"):
        x = batch["input_id"].to(device)
        logits, loss = model(x[:, :-1], x[:, 1:])

        legal = batch["legal"].to(device)
        legal = legal.flatten(-2)[..., all_squares]
        probs = logits.softmax(dim=-1)
        prob_legal = (probs * legal[:, 1:]).sum(dim=-1)
        top1_legal = legal[:, 1:].gather(-1, probs.argmax(dim=-1).unsqueeze(-1)).squeeze(-1)
        avg_prob_legal = prob_legal.mean().item()
        top1_legal_rate = top1_legal.float().mean().item()

        val_loss += loss.item()
        val_prob_legal += avg_prob_legal
        val_top1_legal += top1_legal_rate
        n_batches += 1
    val_loss /= n_batches
    val_prob_legal /= n_batches
    val_top1_legal /= n_batches
    metrics = {
        "val/loss": val_loss,
        "val/prob_legal": val_prob_legal,
        "val/top1_legal": val_top1_legal,
    }
    return metrics

# %%
cfg = GPTConfig(
    vocab_size=32,
    dim=128,
    num_heads=8,
    num_layers=3,
    base=100,
    # n_ctx=31,
)
model = HubGPT(cfg)
num_params = get_num_params(model)
print(f"{num_params} parameters.")
model.to(device)

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=train_cfg.betas)
step = 0
if isinstance(train_cfg.val_steps, float):
    assert isinstance(train_loader.batch_size, int)
    val_steps = train_cfg.val_steps * train_cfg.n_train // train_loader.batch_size
else:
    val_steps = train_cfg.val_steps
val_metrics = {}
if train_cfg.use_wandb:
    wandb.init(project="hyper-connections")
for epoch in range(train_cfg.n_epoch):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for batch in pbar:
        x = batch["input_id"].to(device)
        logits, loss = model(x[:, :-1], x[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        metrics = {"train/loss": loss.item()}
        if step % val_steps == 0:
            val_metrics = eval(model, val_loader)
            metrics.update(val_metrics)

        pbar.set_postfix(metrics | val_metrics | {"step": step})
        if train_cfg.use_wandb:
            wandb.log(metrics, step=step)
if train_cfg.use_wandb:
    wandb.finish()
model.save_pretrained("awonga/othello-gpt-swiglu")

# %%
model = HubHCGPT(cfg)
num_params = get_num_params(model)
print(f"{num_params} parameters.")
model.to(device)

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=train_cfg.betas)
step = 0
if isinstance(train_cfg.val_steps, float):
    assert isinstance(train_loader.batch_size, int)
    val_steps = train_cfg.val_steps * train_cfg.n_train // train_loader.batch_size
else:
    val_steps = train_cfg.val_steps
val_metrics = {}
if train_cfg.use_wandb:
    wandb.init(project="hyper-connections")
for epoch in range(train_cfg.n_epoch):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for batch in pbar:
        x = batch["input_id"].to(device)
        logits, loss = model(x[:, :-1], x[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        metrics = {"train/loss": loss.item()}
        if step % val_steps == 0:
            val_metrics = eval(model, val_loader)
            metrics.update(val_metrics)

        pbar.set_postfix(metrics | val_metrics | {"step": step})
        if train_cfg.use_wandb:
            wandb.log(metrics, step=step)
if train_cfg.use_wandb:
    wandb.finish()
model.save_pretrained("awonga/othello-gpt-hc-swiglu")

# %%
[x for x in model.transformer.A_m]

# %%
[x for x in model.transformer.A_r]

# %%
[x for x in model.transformer.B]

# %%
