# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hyper_connections.model.gpt import HubGPT
from tqdm.auto import tqdm

# %%
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-0724-hf")

# %%
# model = HubGPT(
#     vocab_size=olmo.config.vocab_size,
#     dim=olmo.config.hidden_size,
#     num_heads=olmo.config.num_attention_heads,
#     num_layers=olmo.config.num_hidden_layers,
#     padding_idx=olmo.config.pad_token_id,
#     base=10_000,
# )

# %%
olmo.layers[0].self_attn.q_proj.weight.shape

# %%
{k: v.shape for k, v in olmo.state_dict().items()}

# %%
{k: v.shape for k, v in model.state_dict().items()}

# %%
olmo.model.layers[0].self_attn.rotary_emb

# %%
model.transformer.blocks[0].mlp

# %%
olmo

# %%
model

# %%
