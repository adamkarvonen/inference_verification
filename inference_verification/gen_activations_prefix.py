import os

# Environment setup
os.environ["VLLM_USE_V1"] = "0"

import torch
import pickle
from pathlib import Path
from datetime import datetime
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import string

# Constants
MODEL_NAME = "google/gemma-2-2b-it"
DATASET_NAME = "lmsys/lmsys-chat-1m"
CTX_LEN = 1024
# MAX_DECODE_TOKENS = 256
MAX_DECODE_TOKENS = 128
N_SAMPLES = 10
DTYPE = "bfloat16"
# OUTPUT_DIR = Path(f"activations_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
OUTPUT_DIR = Path("activations")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Activation saving hook
temp_saved_activations = []


def activation_saving_hook(module, input, output):
    temp_saved_activations.append(output[0].detach().clone())


def concatenate_activations(activations_list):
    """Concatenate list of activation tensors along sequence dimension."""
    concatenated = []
    for sample_activations in activations_list:
        concat_tensor = torch.cat(sample_activations, dim=0)
        concatenated.append(concat_tensor)
    return concatenated


def save_data(data, filename):
    """Save data to disk using pickle."""
    filepath = OUTPUT_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {filename}")


def insert_id_tokens(
    prompts: list[list[int]],
    id_tokens: list[int],
    model_name: str,
    insert_pos: int = 4,
    max_len: int | None = None,
) -> list[list[int]]:
    if model_name == "google/gemma-2-2b-it":
        assert insert_pos == 4
    else:
        raise ValueError(f"Model {model_name} not supported")
    out: list[list[int]] = []

    for ids in prompts:
        new_ids = ids[:insert_pos] + id_tokens + ids[insert_pos:]
        print(f"len before: {len(ids)}, len after: {len(new_ids)}")

        # Optional: trim the tail to respect ctx length
        if max_len is not None and len(new_ids) > max_len:
            new_ids = new_ids[:max_len]

        out.append(new_ids)

    return out


def generate_unique_token_ids(
    tokenizer: AutoTokenizer,
) -> tuple[list[list[int]], list[list[int]]]:
    # id_1 = (
    #     "System Prompt: Mention the word apple in your response. Conversation id: "
    #     + "".join(random.choices(string.ascii_letters + string.digits, k=128))
    # )
    # id_2 = (
    #     "System Prompt: Respond as if you are a helpful assistant. Conversation id: "
    #     + "".join(random.choices(string.ascii_letters + string.digits, k=128))
    # )
    id_1 = "Conversation id: " + "".join(
        random.choices(string.ascii_letters + string.digits, k=128)
    )
    id_2 = "Conversation id: " + "".join(
        random.choices(string.ascii_letters + string.digits, k=128)
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


def main():
    # Load tokenizer and dataset
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset(DATASET_NAME, split="train")

    # Prepare prompts
    prompts = [i["conversation"] for _, i in zip(range(N_SAMPLES), ds)]
    prompts = [
        tokenizer.apply_chat_template(prompt, tokenize=False) for prompt in prompts
    ]

    # Tokenize inputs
    tokenized_inputs = tokenizer(
        prompts,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
        truncation=True,
        max_length=CTX_LEN,
    )
    prompt_token_ids = [input_ids for input_ids in tokenized_inputs["input_ids"]]

    id_tokens_1, id_tokens_2 = generate_unique_token_ids(tokenizer)
    prompt_token_ids_a = insert_id_tokens(prompt_token_ids, id_tokens_1, MODEL_NAME)
    prompt_token_ids_b = insert_id_tokens(prompt_token_ids, id_tokens_2, MODEL_NAME)

    # Save original prompts and token IDs
    save_data(
        {
            "prompts": prompts,
            "prompt_token_ids": prompt_token_ids,
            "prompt_token_ids_a": prompt_token_ids_a,
            "prompt_token_ids_b": prompt_token_ids_b,
        },
        "original_data.pkl",
    )

    # Sampling parameters
    generation_params = SamplingParams(
        temperature=0.0, ignore_eos=True, max_tokens=MAX_DECODE_TOKENS + 1
    )
    prefill_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1)

    global temp_saved_activations

    # ======================
    # 16-bit Model Processing
    # ======================
    print("\n" + "=" * 50)
    print("Processing bfloat16 model...")
    print("=" * 50)

    # Load 16-bit model
    llm_16bit = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        max_model_len=CTX_LEN * 2,
        enforce_eager=True,
        dtype=DTYPE,
        disable_async_output_proc=True,
    )
    model_16bit = llm_16bit.llm_engine.model_executor.driver_worker.model_runner.model
    # submodule = model_16bit.model.layers[12]
    submodule = model_16bit.model.norm

    # 16-bit generation
    print("\n16-bit generation phase...")
    saved_activations_handle = submodule.register_forward_hook(activation_saving_hook)
    temp_saved_activations = []
    activations_16bit_gen = []
    tokens_16bit_gen_a = []
    tokens_16bit_gen_b = []

    try:
        for i in tqdm(range(len(prompt_token_ids)), desc="16-bit generation"):
            outputs = llm_16bit.generate(
                prompt_token_ids=prompt_token_ids_a[i],
                sampling_params=generation_params,
                use_tqdm=False,
            )
            activations_16bit_gen.append(temp_saved_activations)
            temp_saved_activations = []
            # Store complete token sequence (prompt + generated)
            generated_tokens = prompt_token_ids_a[i] + list(
                outputs[0].outputs[0].token_ids
            )
            tokens_16bit_gen_a.append(generated_tokens)
            generated_tokens = prompt_token_ids_b[i] + list(
                outputs[0].outputs[0].token_ids
            )
            tokens_16bit_gen_b.append(generated_tokens)
    finally:
        saved_activations_handle.remove()

    # Concatenate activations
    activations_16bit_gen = concatenate_activations(activations_16bit_gen)

    # 16-bit prefill of 16-bit generated tokens
    print("\n16-bit prefill phase (16-bit tokens)...")
    saved_activations_handle = submodule.register_forward_hook(activation_saving_hook)
    temp_saved_activations = []
    activations_16bit_prefill_a = []
    activations_16bit_prefill_b = []
    try:
        for i in tqdm(range(len(tokens_16bit_gen_a)), desc="16-bit prefill"):
            outputs = llm_16bit.generate(
                prompt_token_ids=tokens_16bit_gen_a[i],
                sampling_params=prefill_params,
                use_tqdm=False,
            )
            # Remove last activation (not in original generation)
            activations_16bit_prefill_a.append(temp_saved_activations[0][:-1])
            temp_saved_activations = []
    finally:
        saved_activations_handle.remove()

    saved_activations_handle = submodule.register_forward_hook(activation_saving_hook)
    temp_saved_activations = []
    try:
        for i in tqdm(range(len(tokens_16bit_gen_b)), desc="16-bit prefill"):
            outputs = llm_16bit.generate(
                prompt_token_ids=tokens_16bit_gen_b[i],
                sampling_params=prefill_params,
                use_tqdm=False,
            )
            # Remove last activation (not in original generation)
            activations_16bit_prefill_b.append(temp_saved_activations[0][:-1])
            temp_saved_activations = []
    finally:
        saved_activations_handle.remove()

    # Save 16-bit data
    save_data(
        {
            "activations": activations_16bit_gen,
            "tokens": tokens_16bit_gen_a,
        },
        "16bit_generation.pkl",
    )

    save_data(
        {
            "activations": activations_16bit_prefill_a,
            "tokens": tokens_16bit_gen_a,
        },
        "16bit_prefill_a.pkl",
    )

    save_data(
        {
            "activations": activations_16bit_prefill_b,
            "tokens": tokens_16bit_gen_b,
        },
        "16bit_prefill_b.pkl",
    )

    # Clean up 16-bit model
    del llm_16bit, model_16bit
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 50)
    print("Data collection complete!")
    print(f"All data saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
