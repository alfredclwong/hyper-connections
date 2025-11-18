import torch
from tqdm.auto import tqdm

from hyper_connections.model.gpt import GPT, HCGPT


@torch.no_grad()
def eval(model, tokenized_dataset, device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(tokenized_dataset):
        batch = batch.to(device)
        if isinstance(model, GPT) or isinstance(model, HCGPT):
            _, loss = model(batch[:, :-1], batch[:, 1:])
        else:
            output = model(batch, labels=batch)
            loss = output.loss
        n_batches += 1
        total_loss += loss.item()
    model.train()
    avg_loss = total_loss / n_batches
    return avg_loss
