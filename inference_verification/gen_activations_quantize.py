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

# Constants
MODEL_NAME = "google/gemma-2-2b-it"
DATASET_NAME = "lmsys/lmsys-chat-1m"
CTX_LEN = 1024
MAX_DECODE_TOKENS = 512
# MAX_DECODE_TOKENS = 32
N_SAMPLES = 400
DTYPE = "bfloat16"
# OUTPUT_DIR = Path(f"activations_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
OUTPUT_DIR = Path("activations_quantize")

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

    data_file = "original_data.pkl"
    sixteen_gen_file = "16bit_generation.pkl"
    sixteen_prefill_file = "16bit_prefill.pkl"
    eight_prefill_file = "8bit_prefill_16bit_tokens.pkl"

    # Save original prompts and token IDs
    save_data(
        {
            "prompts": prompts,
            "prompt_token_ids": prompt_token_ids,
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

    if not os.path.exists(OUTPUT_DIR / sixteen_prefill_file):
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
        model_16bit = (
            llm_16bit.llm_engine.model_executor.driver_worker.model_runner.model
        )

        # 16-bit generation
        print("\n16-bit generation phase...")
        saved_activations_handle = model_16bit.model.norm.register_forward_hook(
            activation_saving_hook
        )
        temp_saved_activations = []
        activations_16bit_gen = []
        tokens_16bit_gen = []

        try:
            for i in tqdm(range(len(prompt_token_ids)), desc="16-bit generation"):
                outputs = llm_16bit.generate(
                    prompt_token_ids=prompt_token_ids[i],
                    sampling_params=generation_params,
                    use_tqdm=False,
                )
                activations_16bit_gen.append(temp_saved_activations)
                temp_saved_activations = []
                # Store complete token sequence (prompt + generated)
                generated_tokens = prompt_token_ids[i] + list(
                    outputs[0].outputs[0].token_ids
                )
                tokens_16bit_gen.append(generated_tokens)
        finally:
            saved_activations_handle.remove()

        # Concatenate activations
        activations_16bit_gen = concatenate_activations(activations_16bit_gen)

        # 16-bit prefill of 16-bit generated tokens
        print("\n16-bit prefill phase (16-bit tokens)...")
        saved_activations_handle = model_16bit.model.norm.register_forward_hook(
            activation_saving_hook
        )
        temp_saved_activations = []
        activations_16bit_prefill = []

        try:
            for i in tqdm(range(len(tokens_16bit_gen)), desc="16-bit prefill"):
                outputs = llm_16bit.generate(
                    prompt_token_ids=tokens_16bit_gen[i],
                    sampling_params=prefill_params,
                    use_tqdm=False,
                )
                # Remove last activation (not in original generation)
                activations_16bit_prefill.append(temp_saved_activations[0][:-1])
                temp_saved_activations = []
        finally:
            saved_activations_handle.remove()

        # Save 16-bit data
        save_data(
            {
                "activations": activations_16bit_gen,
                "tokens": tokens_16bit_gen,
            },
            sixteen_gen_file,
        )

        save_data(
            {
                "activations": activations_16bit_prefill,
                "tokens": tokens_16bit_gen,
            },
            sixteen_prefill_file,
        )

        # Clean up 16-bit model
        del llm_16bit, model_16bit
        torch.cuda.empty_cache()
    else:
        print("Skipping 16-bit model processing, data already exists")
        with open(OUTPUT_DIR / sixteen_prefill_file, "rb") as f:
            data = pickle.load(f)
            activations_16bit_prefill = data["activations"]
            tokens_16bit_gen = data["tokens"]

    print("\n" + "=" * 50)
    print("Processing 8-bit quantized model...")
    print("=" * 50)

    # Load 8-bit model
    llm_8bit = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        max_model_len=CTX_LEN * 2,
        enforce_eager=True,
        dtype=DTYPE,
        disable_async_output_proc=True,
        quantization="fp8",
    )
    model_8bit = llm_8bit.llm_engine.model_executor.driver_worker.model_runner.model

    # 8-bit prefill
    print("\n8-bit prefill phase...")
    saved_activations_handle = model_8bit.model.norm.register_forward_hook(
        activation_saving_hook
    )
    temp_saved_activations = []
    activations_8bit_prefill = []

    try:
        for i in tqdm(
            range(len(tokens_16bit_gen)), desc="8-bit prefill of 16-bit tokens"
        ):
            outputs = llm_8bit.generate(
                prompt_token_ids=tokens_16bit_gen[i],
                sampling_params=prefill_params,
                use_tqdm=False,
            )
            # Remove last activation (not in original generation)
            activations_8bit_prefill.append(temp_saved_activations[0][:-1])
            temp_saved_activations = []
    finally:
        saved_activations_handle.remove()

    save_data(
        {
            "activations": activations_8bit_prefill,
            "tokens": tokens_16bit_gen,
        },
        eight_prefill_file,
    )

    # Clean up 8-bit model
    del llm_8bit, model_8bit
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 50)
    print("Data collection complete!")
    print(f"All data saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
