import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import einops
from typing import Optional


def dataset_to_list_of_strs(
    dataset_name: str, min_row_chars: int, total_chars: int
) -> list[str]:
    """
    Grab text data from a streaming dataset, stopping once we've collected total_chars.
    """
    # Adjust column names depending on dataset
    is_redpajama = dataset_name == "togethercomputer/RedPajama-Data-V2"
    # Example for your 'pile' dataset:
    # is_pile = dataset_name == "monology/pile-uncopyrighted"
    column_name = "raw_content" if is_redpajama else "text"

    dataset = load_dataset(
        dataset_name,
        name="sample-10B" if is_redpajama else None,
        trust_remote_code=True,
        streaming=True,
        split="train",
    )

    total_chars_so_far = 0
    result = []

    for row in dataset:
        text = row[column_name]
        if len(text) > min_row_chars:
            result.append(text)
            total_chars_so_far += len(text)
            if total_chars_so_far > total_chars:
                break
    return result


def tokenize_and_concat_dataset(
    tokenizer,
    dataset: list[str],
    seq_len: int,
    add_bos: bool = True,
    max_tokens: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """
    Concatenate text from the dataset with eos_token between chunks, then tokenize.
    Reshape into (B, seq_len) blocks. Truncates any partial remainder.
    """
    full_text = tokenizer.eos_token.join(dataset)

    # Divide into chunks to speed up tokenization
    num_chunks = 20
    chunk_length = (len(full_text) - 1) // num_chunks + 1
    chunks = [
        full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)
    ]
    all_tokens = []
    for chunk in chunks:
        token_ids = tokenizer(chunk)["input_ids"]
        all_tokens.extend(token_ids)

        # Append EOS token if missing.
        if not chunk.endswith(tokenizer.eos_token):
            chunk += tokenizer.eos_token
        token_ids = tokenizer(chunk)["input_ids"]
        all_tokens.extend(token_ids)

    tokens = torch.tensor(all_tokens)

    if max_tokens is not None:
        tokens = tokens[: max_tokens + seq_len + 1]

    num_tokens = len(tokens)
    num_batches = num_tokens // seq_len

    # Drop last partial batch if not full
    tokens = tokens[: num_batches * seq_len]
    tokens = einops.rearrange(
        tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
    )

    # Overwrite first token in each block with BOS if desired
    if add_bos:
        tokens[:, 0] = tokenizer.bos_token_id

    attention_mask = torch.ones_like(tokens)

    token_dict = {
        "input_ids": tokens,
        "attention_mask": attention_mask,
    }

    return token_dict


def load_and_tokenize_and_concat_dataset(
    dataset_name: str,
    ctx_len: int,
    num_tokens: int,
    tokenizer,
    add_bos: bool = True,
    min_row_chars: int = 100,
) -> dict[str, torch.Tensor]:
    """
    Load text from dataset_name, tokenize it, and return (B, ctx_len) blocks of tokens.
    """
    # For safety, let's over-sample from dataset (like you did with 5x)
    dataset_strs = dataset_to_list_of_strs(dataset_name, min_row_chars, num_tokens * 5)

    token_dict = tokenize_and_concat_dataset(
        tokenizer=tokenizer,
        dataset=dataset_strs,
        seq_len=ctx_len,
        add_bos=add_bos,
        max_tokens=num_tokens,
    )

    # Double-check we have enough tokens
    assert (
        token_dict["input_ids"].shape[0] * token_dict["input_ids"].shape[1]
    ) >= num_tokens, "Not enough tokens found!"
    return token_dict
