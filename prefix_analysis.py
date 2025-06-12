# %%
# # ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from __future__ import annotations
import torch
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, List, Tuple
import numpy as np


def make_projection_matrix(
    hidden_dim: int,
    k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Returns an (hidden_dim, k) matrix with columns of unit norm.

    Using a Gaussian i.i.d. matrix works well in practice and
    satisfies the JL lemma with high probability.
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        P = torch.randn(hidden_dim, k, device=device, dtype=dtype, generator=gen)
    else:
        P = torch.randn(hidden_dim, k, device=device, dtype=dtype)

    P /= torch.linalg.norm(P, dim=0, keepdim=True)  # column-wise normalisation
    return P


def compress_matrix_downproj(
    matrix: torch.Tensor,
    P: torch.Tensor,
) -> torch.Tensor:
    """
    Project each token activation into k-dim space.

    matrix: (seq_len, hidden_dim)  on GPU
    P     : (hidden_dim, k)        on same device/dtype
    Returns: (seq_len, k) projected representation
    """
    # matmul is already batched over the sequence axis
    return matrix @ P


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def load_activation_sequences(
    directory: Path, file_map: Dict[str, str]
) -> Dict[str, List[torch.Tensor]]:
    """Load {key: list[Tensor(seq_len, hidden_dim)]} from pickles."""
    result: Dict[str, List[torch.Tensor]] = {}
    for key, filename in file_map.items():
        with open(directory / filename, "rb") as f:
            result[key] = pickle.load(f)["activations"]
    return result


# ------------------------------------------------------------
# Batched compression
# ------------------------------------------------------------
def compress_matrix_topk(matrix: torch.Tensor, k: int) -> Dict[str, torch.Tensor]:
    """
    matrix: (seq_len, hidden_dim) on GPU.
    Returns dict with
        'values'  – (seq_len, k) signed magnitudes
        'indices' – (seq_len, k) positions in hidden-dim
    """
    abs_vals = matrix.abs()  # (L, H)
    top_vals, top_idx = torch.topk(abs_vals, k=k, dim=1)  # (L, k)
    signed_vals = torch.gather(matrix, 1, top_idx)  # preserve sign
    return {"values": signed_vals, "indices": top_idx}


# ------------------------------------------------------------
# Batched –L2 distance for all tokens at once
# ------------------------------------------------------------
def batched_negative_l2(
    comp_a: Dict[str, torch.Tensor], comp_b: Dict[str, torch.Tensor], hidden_dim: int
) -> torch.Tensor:
    """
    Return (seq_len,) tensor of –‖a − b‖₂ for each token.
    Works entirely on GPU.
    """
    seq_len, k = comp_a["values"].shape
    dense_a = torch.zeros((seq_len, hidden_dim), device=comp_a["values"].device)
    dense_b = torch.zeros_like(dense_a)

    row_idx = torch.arange(seq_len, device=dense_a.device).unsqueeze(1).expand(-1, k)
    dense_a[row_idx, comp_a["indices"]] = comp_a["values"]
    dense_b[row_idx, comp_b["indices"]] = comp_b["values"]

    return -torch.norm(dense_a - dense_b, dim=1)  # (seq_len,)


# ------------------------------------------------------------
# Collect per-token scores & labels
# ------------------------------------------------------------
def gather_token_scores(
    triples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int,
    plot_per_triple: bool,
    max_triples: int = 10,
) -> Tuple[List[int], List[float]]:
    """
    Build token-level similarity scores.
    Positive label (1): 16-bit prefill vs 16-bit generation
    Negative label (0): 16-bit prefill vs  8-bit prefill
    """
    y_true, y_scores = [], []

    for i, (seq_prefill16, seq_gen16, seq_prefill8) in enumerate(triples):
        # move to GPU & float32 (fast, deterministic)
        seq_prefill16 = seq_prefill16.float().to(DEVICE, non_blocking=True)
        seq_gen16 = seq_gen16.float().to(DEVICE, non_blocking=True)
        seq_prefill8 = seq_prefill8.float().to(DEVICE, non_blocking=True)

        hidden_dim = seq_prefill16.size(1)
        P = make_projection_matrix(hidden_dim, k, DEVICE, dtype=torch.float32, seed=42)

        # batched compression
        # comp_pre = compress_matrix_topk(seq_prefill16, k)
        # comp_gen = compress_matrix_topk(seq_gen16, k)
        # comp_pre8 = compress_matrix_topk(seq_prefill8, k)
        # # similarity scores for every token
        # pos_scores = batched_negative_l2(comp_pre, comp_gen, hidden_dim)
        # neg_scores = batched_negative_l2(comp_pre, comp_pre8, hidden_dim)

        comp_pre = compress_matrix_downproj(seq_prefill16, P)
        comp_gen = compress_matrix_downproj(seq_gen16, P)
        comp_pre8 = compress_matrix_downproj(seq_prefill8, P)

        pos_scores = -torch.norm(comp_pre - comp_gen, dim=1)  # (seq_len,)
        neg_scores = -torch.norm(comp_pre - comp_pre8, dim=1)

        if plot_per_triple:
            if i >= max_triples:
                break
            plt.figure(figsize=(8, 4))
            plt.plot(
                pos_scores.float().cpu().numpy(),
                label="pos (prefill16 vs gen16)",
                color="blue",
            )
            plt.plot(
                neg_scores.float().cpu().numpy(),
                label="neg (prefill16 vs prefill8)",
                color="red",
            )
            plt.title(f"Token Scores for Triple {i + 1} (k={k})")
            plt.xlabel("Token index")
            plt.ylabel("Score (-L2 distance)")
            plt.ylim(-2, 0)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # extend python lists (detach→cpu to avoid GPU <--> CPU chatter later)
        y_scores.extend(pos_scores.detach().cpu().tolist())
        y_true.extend([1] * len(pos_scores))

        y_scores.extend(neg_scores.detach().cpu().tolist())
        y_true.extend([0] * len(neg_scores))

    return y_true, y_scores


# ------------------------------------------------------------
# PR plotting
# ------------------------------------------------------------
def plot_pr_curves(
    k_values: List[int],
    seq_triples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    plot_per_triple: bool,
) -> None:
    plt.figure(figsize=(7, 5))
    for k in k_values:
        labels, scores = gather_token_scores(seq_triples, k, plot_per_triple)
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"k={k} (AUC={pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Token-Level Activation Similarity\nPrecision–Recall")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
if __name__ == "__main__":
    DEVICE = torch.device("cuda")  # ← run on GPU
    ACTIVATION_DIR = Path("activations")
    # ACTIVATION_DIR = Path("activations_prefix_longer")
    FILES = {
        "prefill16": "16bit_prefill_a.pkl",
        "gen16": "16bit_generation.pkl",
        "prefill8": "16bit_prefill_b.pkl",
    }

    # ACTIVATION_DIR = Path("activations_v2")
    # FILES = {
    #     "prefill16": "16bit_prefill.pkl",
    #     "gen16": "16bit_generation.pkl",
    #     "prefill8": "8bit_prefill_16bit_tokens.pkl",
    # }
    K_VALUES: List[int] = [4, 32, 64, 128, 256]
    K_VALUES = [32]
    plot_per_triple = True

    data = load_activation_sequences(ACTIVATION_DIR, FILES)
    sequence_triples = list(zip(data["prefill16"], data["gen16"], data["prefill8"]))

    plot_pr_curves(K_VALUES, sequence_triples, plot_per_triple)
# %%
