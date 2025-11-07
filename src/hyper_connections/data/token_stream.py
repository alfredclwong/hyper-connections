from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import torch


class TokenizedDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, tokenizer: AutoTokenizer, batch_size: int, seq_len: int, n_tokens: int | None = None, drop_last: bool = True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = seq_len
        self.n_tokens = n_tokens if n_tokens is not None else None
        self.drop_last = drop_last
        self.batch_size = batch_size

    def __iter__(self):
        buffer = []
        tokens_yielded = 0

        for example in self.dataset:
            tokens = self.tokenizer.encode(example["text"], add_special_tokens=False)
            tokens += [self.tokenizer.eos_token_id]

            # Add tokens to the buffer
            buffer.extend(tokens)

            # While we have enough tokens, yield a full sequence
            while len(buffer) >= self.max_seq_len * self.batch_size:
                if self.n_tokens is not None and tokens_yielded >= self.n_tokens:
                    return

                batch = buffer[:self.max_seq_len * self.batch_size]
                batch = torch.tensor(batch, dtype=torch.long)
                batch = batch.view(self.batch_size, self.max_seq_len)
                # batch = batch.to(self.device)
                yield batch

                buffer = buffer[self.max_seq_len * self.batch_size:]  # keep the overflow
                tokens_yielded += self.max_seq_len * self.batch_size

        # Optionally, drop or yield the last partial sequence
        if not self.drop_last and len(buffer) > 0:
            if self.n_tokens is not None and tokens_yielded >= self.n_tokens:
                return

            # Pad the last batch if necessary
            while len(buffer) < self.max_seq_len * self.batch_size:
                buffer.append(self.tokenizer.pad_token_id)

            batch = torch.tensor(buffer[:self.max_seq_len * self.batch_size], dtype=torch.long)
            batch = batch.view(self.batch_size, self.max_seq_len)
            batch = batch.to(self.device)
            yield batch        

    def __len__(self):
        if self.n_tokens is None:
            return None
        length = self.n_tokens // self.max_seq_len // self.batch_size
        if not self.drop_last and self.n_tokens % self.max_seq_len != 0:
            length += 1
        return length
