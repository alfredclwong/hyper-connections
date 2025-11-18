import torch
from transformers import AutoModelForCausalLM

from hyper_connections.model.gpt import HubGPT
from hyper_connections.model.hc import HCNet


def view_HC(model: HCNet):
    L = len(model.blocks)
    N = model.n
    if N > 0:
        HC = torch.zeros(L, N + 1, N + 1)
        HC[:, [0], 1:] = model.B.detach().cpu()
        HC[:, 1:, [0]] = model.A_m.detach().cpu()
        HC[:, 1:, 1:] = model.A_r.detach().cpu()
    else:
        HC = torch.zeros(L, -N + 1, -N * 2)
        HC[:, [0], -N:] = model.B.detach().cpu()
        HC[:, 1:, :-N] = model.A_m.detach().cpu()
        HC[:, 1:, -N:] = model.A_r.detach().cpu()
    return HC


def get_num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_max_memory_usage(model: torch.nn.Module, batch_size: int) -> float:
    num_params = get_num_params(model)
    bytes_per_param = 4  # assuming float32
    multiplier = 4  # params, gradients, optimizer states (momentum, variance)
    mb = batch_size * num_params * bytes_per_param * multiplier / (1024**2)
    return mb


def get_device():
    return (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
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
