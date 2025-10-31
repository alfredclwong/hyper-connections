# %%
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from hyper_connections.model.gpt import HubGPT
from hyper_connections.model.hc import HubHCGPT
from tqdm.auto import tqdm

# %%
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
dataset = load_dataset("allenai/dolma", split="train", streaming=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = 1
tokenizer.unk_token_id = 1

# %%
tokenizer.special_tokens_map

# %%
model = HubGPT(
# model = HubHCGPT(
    vocab_size=tokenizer.vocab_size,
    dim=768,
    num_heads=12,
    num_layers=12,
    base=10_000,
)
model.to(device)

# %%
train_loader = DataLoader(dataset, batch_size=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-2, betas=(0.9, 0.95))
model.train()
n_train = 100_000
pbar = tqdm(train_loader, total=n_train)
for i, batch in enumerate(pbar):
    inputs = tokenizer(batch["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    optimizer.zero_grad()
    logits, loss = model(input_ids[:, :-1], y=input_ids[:, 1:])
    loss.backward()
    optimizer.step()

    pbar.set_description(f"Loss: {loss.item():.4f}")

    if i >= n_train - 1:
        break

# %%
(
    torch.stack([model.transformer.A_m[i] for i in range(len(model.transformer.blocks))]),
    torch.stack([model.transformer.A_r[i] for i in range(len(model.transformer.blocks))]),
    torch.stack([model.transformer.B[i] for i in range(len(model.transformer.blocks))]),
)

# %%
def generate_topk(model, input_ids, top_k=5, max_length=50):
    model.eval()
    generated = input_ids
    with torch.no_grad():
        for _ in tqdm(range(max_length)):
            logits, _ = model(generated)
            next_token_logits = logits[:, -1, :]
            topk_probs, topk_indices = torch.topk(torch.softmax(next_token_logits, dim=-1), top_k, dim=-1)
            next_token_id = torch.multinomial(topk_probs, num_samples=1)
            next_token_id = torch.gather(topk_indices, -1, next_token_id)
            generated = torch.cat([generated, next_token_id], dim=-1)
    return generated

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in tqdm(range(max_new_tokens)):
        # forward the model to get the logits for the index in the sequence
        output = model(idx)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output.logits
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = logits.softmax(dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# %%
model.eval()
n_test = 200
total_loss = 0.0
pbar = tqdm(train_loader, total=n_test)
with torch.no_grad():
    for i, batch in enumerate(pbar):
        inputs = tokenizer(batch["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        logits, loss = model(input_ids[:, :-1], y=input_ids[:, 1:])
        total_loss += loss.item()

        pbar.set_description(f"Avg Loss: {total_loss / (i + 1):.4f}")

        for _ in tqdm(range(50)):
            logits, _ = model(input_ids)
            next_token_id = logits.argmax(dim=-1)[:, -1:]
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        print(tokenizer.decode(input_ids[0].tolist()))

        break

        if i >= n_test - 1:
            break

# %%
sample = "In a distant future, "
input_ids = tokenizer(sample, return_tensors="pt")["input_ids"].to(device)
generated = generate(model, input_ids, max_new_tokens=40, top_k=50)
print(tokenizer.decode(generated[0].tolist()))

# %%
def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params(gpt2), num_params(model)

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %%
sample = "In a distant future, "
model_inputs = gpt2_tokenizer(sample, return_tensors="pt").to(device)
# generated = gpt2.generate(
#     **model_inputs,
#     max_new_tokens=40,
#     do_sample=True,
#     top_k=50,
#     top_p=0.95,
#     num_return_sequences=3,
# )
generated = generate(gpt2, model_inputs["input_ids"], max_new_tokens=40, top_k=50)
print(gpt2_tokenizer.decode(generated[0].tolist()))

# %%
