"""
Standalone validation: v3 (best-first) vs v4 (prefix-aware greedy).

Generates synthetic draft logits under controlled distributions, builds
trees with both algorithms, and computes *exact* E[tau] for each tree.
No GPU required — runs on CPU in seconds.

Usage:
    python test_v4_validation.py
"""

import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import torch

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "dflash_tree", os.path.join(os.path.dirname(__file__), "model", "dflash_tree.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_bestfirst_tree = _mod.build_bestfirst_tree
build_prefixaware_tree = _mod.build_prefixaware_tree


def compute_exact_expected_tau(
    leaf_tokens: List[List[int]],
    topk_logprobs_cpu: List[List[float]],
    topk_tokens_cpu: List[List[int]],
    seq_len: int,
) -> Tuple[float, List[float]]:
    """
    Compute exact E[tau] = sum_{k=1}^{D} P(tau >= k) for a set of leaves.

    P(tau >= k) = sum over distinct depth-k prefixes sigma of P_dft(sigma),
    where P_dft(sigma) = prod_{d=1}^{k} p_d(sigma_d).

    Returns (E_tau, [P(tau>=1), P(tau>=2), ...]).
    """
    token_to_logprob: List[Dict[int, float]] = []
    for d in range(seq_len):
        mapping = {}
        for j, tok in enumerate(topk_tokens_cpu[d]):
            mapping[tok] = topk_logprobs_cpu[d][j]
        token_to_logprob.append(mapping)

    depth_prefixes: List[Set[tuple]] = [set() for _ in range(seq_len)]
    for toks in leaf_tokens:
        real_toks = [t for t in toks if t != -1]
        for k in range(len(real_toks)):
            depth_prefixes[k].add(tuple(real_toks[: k + 1]))

    p_tau_geq = []
    for k in range(seq_len):
        prob = 0.0
        for prefix in depth_prefixes[k]:
            clp = 0.0
            valid = True
            for d, tok in enumerate(prefix):
                if tok in token_to_logprob[d]:
                    clp += token_to_logprob[d][tok]
                else:
                    valid = False
                    break
            if valid:
                prob += math.exp(clp)
        p_tau_geq.append(prob)

    e_tau = sum(p_tau_geq)
    return e_tau, p_tau_geq


def make_logits(probs_per_pos: List[List[float]], vocab_size: int = 32) -> torch.Tensor:
    """
    Build [1, seq_len, vocab_size] logits tensor from per-position probability
    lists.  probs_per_pos[d] gives probs for the first len(probs) tokens at
    depth d; remaining vocab gets uniform leftover mass.
    """
    seq_len = len(probs_per_pos)
    logits = torch.full((1, seq_len, vocab_size), -10.0)
    for d, probs in enumerate(probs_per_pos):
        for j, p in enumerate(probs):
            logits[0, d, j] = math.log(p + 1e-30)
    return logits


def run_scenario(
    name: str,
    probs_per_pos: List[List[float]],
    max_tree_size: int,
    expand_k: int,
    vocab_size: int = 32,
):
    """Build trees with v3 and v4, compute and compare exact E[tau]."""
    logits = make_logits(probs_per_pos, vocab_size)
    seq_len = len(probs_per_pos)
    anchor = torch.tensor([[0]], dtype=torch.long)

    _, _, _, leaf_paths_v3, leaf_tokens_v3 = build_bestfirst_tree(
        logits, anchor, max_tree_size=max_tree_size, expand_k=expand_k,
    )
    _, _, _, leaf_paths_v4, leaf_tokens_v4 = build_prefixaware_tree(
        logits, anchor, max_tree_size=max_tree_size, expand_k=expand_k,
    )

    lv3 = leaf_tokens_v3.tolist()
    lv4 = leaf_tokens_v4.tolist()

    log_probs_all = logits[0] - torch.logsumexp(logits[0], dim=-1, keepdim=True)
    topk_lp, topk_idx = torch.topk(log_probs_all, k=expand_k, dim=-1)
    topk_lp_cpu = topk_lp.tolist()
    topk_idx_cpu = topk_idx.tolist()

    # For E[tau] we need the full token->logprob mapping (not just top-K).
    # Rebuild from the raw log_probs_all for all tokens that appear in leaves.
    full_lp = log_probs_all.tolist()
    full_tok = list(range(vocab_size))
    full_topk_lp = [[full_lp[d][t] for t in range(vocab_size)] for d in range(seq_len)]
    full_topk_tok = [list(range(vocab_size)) for _ in range(seq_len)]

    e3, ptau3 = compute_exact_expected_tau(lv3, full_topk_lp, full_topk_tok, seq_len)
    e4, ptau4 = compute_exact_expected_tau(lv4, full_topk_lp, full_topk_tok, seq_len)

    delta = e4 - e3
    pct = (delta / e3 * 100) if e3 > 0 else 0.0
    winner = "v4" if delta > 1e-6 else ("v3" if delta < -1e-6 else "tie")

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  seq_len={seq_len}  max_tree_size={max_tree_size}  expand_k={expand_k}")
    print(f"{'='*70}")
    print(f"  v3 leaves ({len(lv3)}):")
    for toks in lv3:
        real = [t for t in toks if t != -1]
        print(f"    {real}")
    print(f"  v4 leaves ({len(lv4)}):")
    for toks in lv4:
        real = [t for t in toks if t != -1]
        print(f"    {real}")
    print()
    for k in range(seq_len):
        print(f"  P(tau>={k+1}):  v3={ptau3[k]:.6f}   v4={ptau4[k]:.6f}   diff={ptau4[k]-ptau3[k]:+.6f}")
    print(f"  -----------------------------------------")
    print(f"  E[tau]:       v3={e3:.6f}   v4={e4:.6f}   diff={delta:+.6f} ({pct:+.1f}%)  winner={winner}")

    return e3, e4


def main():
    print("=" * 70)
    print("  v3 (best-first) vs v4 (prefix-aware greedy) — exact E[tau]")
    print("=" * 70)

    results = {}

    # ------------------------------------------------------------------
    # Scenario 1: Near-uniform distribution (counterexample regime).
    # v3 concentrates on one prefix branch; v4 should diversify.
    # ------------------------------------------------------------------
    results["uniform"] = run_scenario(
        name="Scenario 1: Near-uniform (high entropy per position)",
        probs_per_pos=[
            [0.40, 0.35, 0.25],
            [0.40, 0.35, 0.25],
            [0.40, 0.35, 0.25],
        ],
        max_tree_size=4,
        expand_k=3,
        vocab_size=8,
    )

    # ------------------------------------------------------------------
    # Scenario 2: Peaked distribution (v3's strength).
    # One dominant token per position; prefix diversity less important.
    # ------------------------------------------------------------------
    results["peaked"] = run_scenario(
        name="Scenario 2: Peaked (low entropy, one dominant token)",
        probs_per_pos=[
            [0.90, 0.05, 0.05],
            [0.85, 0.10, 0.05],
            [0.80, 0.10, 0.10],
        ],
        max_tree_size=4,
        expand_k=3,
        vocab_size=8,
    )

    # ------------------------------------------------------------------
    # Scenario 3: Mixed — peaked at pos 1, uniform at pos 2-3.
    # ------------------------------------------------------------------
    results["mixed"] = run_scenario(
        name="Scenario 3: Mixed (peaked d=1, uniform d=2-3)",
        probs_per_pos=[
            [0.85, 0.10, 0.05],
            [0.40, 0.35, 0.25],
            [0.40, 0.35, 0.25],
        ],
        max_tree_size=4,
        expand_k=3,
        vocab_size=8,
    )

    # ------------------------------------------------------------------
    # Scenario 4: Larger tree — D=5, M=8, K=3.
    # ------------------------------------------------------------------
    results["deep"] = run_scenario(
        name="Scenario 4: Deeper tree (D=5, M=8, K=3, near-uniform)",
        probs_per_pos=[
            [0.40, 0.35, 0.25],
            [0.40, 0.35, 0.25],
            [0.40, 0.35, 0.25],
            [0.40, 0.35, 0.25],
            [0.40, 0.35, 0.25],
        ],
        max_tree_size=8,
        expand_k=3,
        vocab_size=8,
    )

    # ------------------------------------------------------------------
    # Scenario 5: Realistic-ish — D=4, M=16, K=4.
    # ------------------------------------------------------------------
    results["realistic"] = run_scenario(
        name="Scenario 5: Realistic-ish (D=4, M=16, K=4)",
        probs_per_pos=[
            [0.50, 0.20, 0.15, 0.15],
            [0.45, 0.25, 0.20, 0.10],
            [0.55, 0.20, 0.15, 0.10],
            [0.60, 0.15, 0.15, 0.10],
        ],
        max_tree_size=16,
        expand_k=4,
        vocab_size=16,
    )

    # ------------------------------------------------------------------
    # Scenario 6: Very peaked (v3 ≈ v4, both near-optimal).
    # ------------------------------------------------------------------
    results["very_peaked"] = run_scenario(
        name="Scenario 6: Very peaked (v3 should ≈ v4)",
        probs_per_pos=[
            [0.95, 0.03, 0.02],
            [0.93, 0.04, 0.03],
            [0.90, 0.06, 0.04],
            [0.92, 0.05, 0.03],
        ],
        max_tree_size=8,
        expand_k=3,
        vocab_size=8,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    v4_wins = 0
    v3_wins = 0
    ties = 0
    for name, (e3, e4) in results.items():
        delta = e4 - e3
        pct = (delta / e3 * 100) if e3 > 0 else 0.0
        tag = "v4 wins" if delta > 1e-6 else ("v3 wins" if delta < -1e-6 else "tie")
        if delta > 1e-6:
            v4_wins += 1
        elif delta < -1e-6:
            v3_wins += 1
        else:
            ties += 1
        print(f"  {name:20s}  v3={e3:.4f}  v4={e4:.4f}  diff={delta:+.4f} ({pct:+.1f}%)  {tag}")
    print(f"\n  v4 wins: {v4_wins}   v3 wins: {v3_wins}   ties: {ties}")
    print()

    if v3_wins > v4_wins:
        print("  WARNING: v4 underperforms v3 in more scenarios than expected.")
        print("  Check the algorithm implementation.")
        sys.exit(1)
    else:
        print("  PASS: v4 >= v3 in the majority of scenarios.")
        sys.exit(0)


if __name__ == "__main__":
    main()
