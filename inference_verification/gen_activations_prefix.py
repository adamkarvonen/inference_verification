import os

os.environ.setdefault("VLLM_USE_V1", "0")

from pathlib import Path
from datetime import datetime
import pickle
from typing import List, Tuple

import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import string

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def save_pickle(obj, filename: str) -> None:
    path = OUTPUT_DIR / filename
    with path.open("wb") as f:
        pickle.dump(obj, f)
    print(f"[+] Saved {path}")


def generate_unique_ids(
    tokenizer: AutoTokenizer, num_chars: int = 64
) -> tuple[list[list[int]], list[list[int]]]:
    id_1 = "Conversation id: " + "".join(
        random.choices(string.ascii_letters + string.digits, k=num_chars)
    )
    id_2 = "Conversation id: " + "".join(
        random.choices(string.ascii_letters + string.digits, k=num_chars)
    )

    print(id_1)
    print(id_2)

    id_tokens_1 = tokenizer.encode(id_1, add_special_tokens=False, return_tensors=None)
    id_tokens_2 = tokenizer.encode(id_2, add_special_tokens=False, return_tensors=None)

    min_length = min(len(id_tokens_1), len(id_tokens_2))

    id_tokens_1 = id_tokens_1[:min_length] + tokenizer.encode(
        "\n", add_special_tokens=False
    )
    id_tokens_2 = id_tokens_2[:min_length] + tokenizer.encode(
        "\n", add_special_tokens=False
    )

    return id_tokens_1, id_tokens_2


def insert_ids(
    seq: List[int], ids: List[int], *, max_len: int | None = None
) -> List[int]:
    # keep first INSERT_POS tokens, insert ids, append the rest
    out = seq[:INSERT_POS] + ids + seq[INSERT_POS:]
    return out[:max_len] if max_len and len(out) > max_len else out


class ActivationCatcher:
    """Context manager that records activations from a module."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.saved: list[torch.Tensor] = []
        self._handle = None

    def __enter__(self):
        self._handle = self.module.register_forward_hook(
            lambda m, inp, out: self.saved.append(out[0].detach().clone())
        )
        return self.saved

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()


def concatenate_batch(batch: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    return [torch.cat(sample, dim=0) for sample in batch]


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Tokenizer and data
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset(DATASET_NAME, split="train")

    # prepare raw prompts (chat template -> text -> token ids)
    raw_prompts = [ds[i]["conversation"] for i in range(N_SAMPLES)]
    rendered_prompts = [tok.apply_chat_template(p, tokenize=False) for p in raw_prompts]
    tokenized = tok(
        rendered_prompts, add_special_tokens=False, truncation=True, max_length=CTX_LEN
    )
    prompt_ids: list[list[int]] = tokenized["input_ids"]

    # splice in unique conversation ids
    id_a, id_b = generate_unique_ids(tok)
    prompts_a = [insert_ids(ids, id_a, max_len=CTX_LEN) for ids in prompt_ids]
    prompts_b = [insert_ids(ids, id_b, max_len=CTX_LEN) for ids in prompt_ids]

    # store references for reproducibility
    save_pickle(
        {
            "rendered_prompts": rendered_prompts,
            "prompt_ids": prompt_ids,
            "prompts_a": prompts_a,
            "prompts_b": prompts_b,
        },
        "original_data.pkl",
    )

    # instantiate model once (bfloat16)
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        max_model_len=CTX_LEN * 2,
        enforce_eager=True,
        dtype=DTYPE,
        disable_async_output_proc=True,
    )
    model_core = llm.llm_engine.model_executor.driver_worker.model_runner.model

    generation_cfg = SamplingParams(
        temperature=0.0, ignore_eos=True, max_tokens=MAX_DECODE_TOKENS + 1
    )
    prefill_cfg = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1)

    # --- Phase 1: generation --------------------------------------------------
    print("[Phase 1] Autoregressive generation …")
    activations_gen, tokens_gen_a, tokens_gen_b = [], [], []
    for seq_ids in tqdm(prompts_a, desc="generate A & B"):
        with ActivationCatcher(model_core.model.norm) as acts:
            out = llm.generate(
                prompt_token_ids=seq_ids, sampling_params=generation_cfg, use_tqdm=False
            )
        activations_gen.append(acts)
        # save full sequences (prompt + generated) for both A and B versions
        gen_toks = list(out[0].outputs[0].token_ids)
        tokens_gen_a.append(seq_ids + gen_toks)
        # reuse B prompt for the same generated tail
        idx = prompts_a.index(seq_ids)
        tokens_gen_b.append(prompts_b[idx] + gen_toks)

    save_pickle(
        {
            "activations": concatenate_batch(activations_gen),
            "tokens": tokens_gen_a,
            "ids_a": id_a,
            "ids_b": id_b,
        },
        "16bit_generation.pkl",
    )

    # --- Phase 2: prefill ------------------------------------------------------
    def prefill(tokens: list[list[int]], tag: str):
        res = []
        for seq in tqdm(tokens, desc=f"prefill {tag}"):
            with ActivationCatcher(model_core.model.norm) as acts:
                _ = llm.generate(
                    prompt_token_ids=seq, sampling_params=prefill_cfg, use_tqdm=False
                )
            # drop final step (not part of prompt)
            res.append(acts[0][:-1])
        save_pickle({"activations": res, "tokens": tokens}, f"16bit_prefill_{tag}.pkl")

    prefill(tokens_gen_a, "a")
    prefill(tokens_gen_b, "b")

    print("\n[✓] Data collection complete →", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    MODEL_NAME = "google/gemma-2-2b-it"
    DATASET_NAME = "lmsys/lmsys-chat-1m"
    CTX_LEN = 1_024
    MAX_DECODE_TOKENS = 512
    N_SAMPLES = 400
    DTYPE = "bfloat16"  # "float16" also works
    OUTPUT_DIR = Path("activations_prefix")
    INSERT_POS = 4  # works for Gemma; adjust for other models

    main()
