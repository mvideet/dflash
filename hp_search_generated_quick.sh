#!/bin/bash
# Auto-generated hp_search script - run with: bash hp_search_generated.sh
# Note: no set -e so benchmark failures don't stop the search
cd "$(dirname "$0")"
mkdir -p logs
RESULTS=logs/hp_search_results.csv
echo 'block_size,max_tree_size,theta_uni,theta_bi,theta_tri,prune_top_k,avg_acceptance,speedup' > "$RESULTS"

echo '[1/8] block_size=8 max_tree_size=4 theta=(0.85,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 /homes/videetm/miniforge3/envs/dflash/bin/torchrun --nproc_per_node=8 --master_port=29600 benchmark.py --dataset mt-bench --max-samples 4 --max-new-tokens 128 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --block-size 8 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,4,0.85,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,4,0.85,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[2/8] block_size=8 max_tree_size=4 theta=(0.9,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 /homes/videetm/miniforge3/envs/dflash/bin/torchrun --nproc_per_node=8 --master_port=29601 benchmark.py --dataset mt-bench --max-samples 4 --max-new-tokens 128 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --block-size 8 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,4,0.9,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,4,0.9,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[3/8] block_size=8 max_tree_size=6 theta=(0.85,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 /homes/videetm/miniforge3/envs/dflash/bin/torchrun --nproc_per_node=8 --master_port=29602 benchmark.py --dataset mt-bench --max-samples 4 --max-new-tokens 128 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --block-size 8 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,6,0.85,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,6,0.85,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[4/8] block_size=8 max_tree_size=6 theta=(0.9,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 /homes/videetm/miniforge3/envs/dflash/bin/torchrun --nproc_per_node=8 --master_port=29603 benchmark.py --dataset mt-bench --max-samples 4 --max-new-tokens 128 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --block-size 8 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,6,0.9,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,6,0.9,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[5/8] block_size=16 max_tree_size=4 theta=(0.85,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 /homes/videetm/miniforge3/envs/dflash/bin/torchrun --nproc_per_node=8 --master_port=29604 benchmark.py --dataset mt-bench --max-samples 4 --max-new-tokens 128 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --block-size 16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 16,4,0.85,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "16,4,0.85,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[6/8] block_size=16 max_tree_size=4 theta=(0.9,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 /homes/videetm/miniforge3/envs/dflash/bin/torchrun --nproc_per_node=8 --master_port=29605 benchmark.py --dataset mt-bench --max-samples 4 --max-new-tokens 128 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --block-size 16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 16,4,0.9,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "16,4,0.9,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[7/8] block_size=16 max_tree_size=6 theta=(0.85,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 /homes/videetm/miniforge3/envs/dflash/bin/torchrun --nproc_per_node=8 --master_port=29606 benchmark.py --dataset mt-bench --max-samples 4 --max-new-tokens 128 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --block-size 16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 16,6,0.85,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "16,6,0.85,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[8/8] block_size=16 max_tree_size=6 theta=(0.9,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 /homes/videetm/miniforge3/envs/dflash/bin/torchrun --nproc_per_node=8 --master_port=29607 benchmark.py --dataset mt-bench --max-samples 4 --max-new-tokens 128 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --block-size 16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 16,6,0.9,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "16,6,0.9,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"
