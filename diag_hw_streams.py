"""HW Idea 6 feasibility: can CUDA streams overlap draft + verify on a single GPU?

The full Idea 6 needs 2 GPUs + NVLink. A cheap proxy: same GPU, two streams.
If draft + verify can overlap on the SM scheduler (SMs not saturated by either
alone), wall time should drop. If both saturate the GPU, no overlap possible
and Idea 6 needs multi-GPU.

Test: simulate concurrent draft + verify forward passes via cuda.streams.
Compare to sequential.
"""
import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM
from model import DFlashDraftModel
from transformers import DynamicCache

DEVICE = torch.device("cuda:0")


def main():
    print("Loading models...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    draft = DFlashDraftModel.from_pretrained(
        "z-lab/Qwen3-4B-DFlash-b16", attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    block_size = draft.block_size

    B, M = 4, 32
    prefix_len = 200

    # Pre-fill target KV
    target_kv = DynamicCache()
    pre = torch.randint(0, 50000, (B, prefix_len), device=DEVICE)
    pre_attn = torch.ones(B, prefix_len, dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        target.model(pre, attention_mask=pre_attn, past_key_values=target_kv,
                     use_cache=True)

    # Verify input
    verify_input = torch.randint(0, 50000, (B, M), device=DEVICE)
    verify_pos = torch.arange(prefix_len, prefix_len + M, device=DEVICE).unsqueeze(0).expand(B, -1)
    verify_attn = torch.zeros(B, 1, M, prefix_len + M, dtype=torch.bfloat16, device=DEVICE)

    # Draft input (synthetic — match interface)
    draft_target_hidden = torch.randn(B, prefix_len, draft.config.hidden_size,
                                       dtype=torch.bfloat16, device=DEVICE)
    noise_emb = target.model.embed_tokens(
        torch.randint(0, 50000, (B, block_size), device=DEVICE))
    draft_pos = torch.arange(prefix_len, prefix_len + block_size, device=DEVICE).unsqueeze(0).expand(B, -1)
    draft_kv = DynamicCache()

    import copy
    target_kv_snap = copy.deepcopy(target_kv)
    draft_kv_snap = copy.deepcopy(draft_kv)

    def run_verify():
        kv = copy.deepcopy(target_kv_snap)
        with torch.inference_mode():
            return target.model(verify_input, position_ids=verify_pos,
                                past_key_values=kv, use_cache=False,
                                attention_mask=verify_attn)

    # Synthetic "draft" — just a target.model call at smaller q_len
    # to simulate the per-call cost of the drafter.
    draft_input = torch.randint(0, 50000, (B, block_size), device=DEVICE)
    draft_attn = torch.zeros(B, 1, block_size, prefix_len + block_size,
                             dtype=torch.bfloat16, device=DEVICE)
    def run_draft():
        kv = copy.deepcopy(target_kv_snap)
        with torch.inference_mode():
            return target.model(draft_input, position_ids=draft_pos[:, :block_size],
                                past_key_values=kv, use_cache=False,
                                attention_mask=draft_attn)

    # Time sequential
    for _ in range(3): run_verify(); run_draft()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(15):
        run_verify()
        run_draft()
    torch.cuda.synchronize()
    seq_ms = (time.perf_counter() - t0) / 15 * 1000
    print(f"Sequential verify+draft: {seq_ms:.2f} ms/iter")

    # Time concurrent via streams
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    for _ in range(3):
        with torch.cuda.stream(s1): run_verify()
        with torch.cuda.stream(s2): run_draft()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(15):
        with torch.cuda.stream(s1): run_verify()
        with torch.cuda.stream(s2): run_draft()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    par_ms = (time.perf_counter() - t0) / 15 * 1000
    print(f"Parallel  verify+draft: {par_ms:.2f} ms/iter")

    overlap_pct = (1 - par_ms / seq_ms) * 100
    print(f"\nOverlap savings: {overlap_pct:.1f}% (vs theoretical max where both fully overlap)")
    print(f"  → Idea 6 single-GPU ceiling: {overlap_pct:.1f}%")
    if overlap_pct < 5:
        print(f"  → GPU saturated by either pass alone; multi-GPU needed for real Idea 6 gain")
    elif overlap_pct < 20:
        print(f"  → Modest overlap available; multi-GPU would be cleaner")
    else:
        print(f"  → Real overlap possible single-GPU; multi-GPU for further gain")

    out = {"B": B, "M": M, "prefix": prefix_len,
           "sequential_ms": seq_ms, "parallel_ms": par_ms,
           "overlap_pct": overlap_pct}
    os.makedirs("logs", exist_ok=True)
    with open("logs/hw_streams.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote logs/hw_streams.json")


if __name__ == "__main__":
    main()
