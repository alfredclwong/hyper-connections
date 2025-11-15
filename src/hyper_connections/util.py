import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyper_connections.data.token_stream import TokenizedDataset
from hyper_connections.model.gpt import HubGPT


def get_num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_device():
    return (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf", pad_token_id=1)
    return tokenizer


def get_tokenized_dataset(tokenizer, n_tokens, batch_size, seq_len, shuffle=False, **dataset_kwargs):
    streaming_dataset = load_dataset(**dataset_kwargs)
    if shuffle:
        streaming_dataset = streaming_dataset.shuffle(buffer_size=10_000, seed=42)
    tokenized_dataset = TokenizedDataset(
        dataset=streaming_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        n_tokens=n_tokens,
        drop_last=True,
    )
    return tokenized_dataset


def get_tokenized_dolma_train_dataset(tokenizer, n_tokens, batch_size, seq_len):
    dolma_train_kwargs = {
        "path": "allenai/dolma",
        "name": "v1_5-sample",
        "split": "train",
        "streaming": True,
        "trust_remote_code": True,
    }
    return get_tokenized_dataset(
        tokenizer,
        n_tokens,
        batch_size,
        seq_len,
        shuffle=True,
        **dolma_train_kwargs,
    )


def get_tokenized_c4_val_dataset(tokenizer, n_tokens, batch_size, seq_len):
    c4_val_kwargs = {
        "path": "allenai/c4",
        "name": "en",
        "split": "validation",
        "streaming": True,
        "trust_remote_code": True,
    }
    return get_tokenized_dataset(
        tokenizer,
        n_tokens,
        batch_size,
        seq_len,
        **c4_val_kwargs,
    )


def get_model(device):
    model = HubGPT.from_pretrained(
        "awonga/HubGPT-ckpt10212",
        vocab_size=51200,
        dim=768,
        num_heads=12,
        num_layers=12,
        base=10_000,
    )
    model.to(device)
    return model


def get_olmo(device):
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-0724-hf")
    model.to(device)
    return model
