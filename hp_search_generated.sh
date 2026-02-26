#!/bin/bash
# Auto-generated hp_search script - run with: bash hp_search_generated.sh
# Note: no set -e so benchmark failures don't stop the search
cd "$(dirname "$0")"
mkdir -p logs
RESULTS=logs/hp_search_results.csv
echo 'max_tree_size,theta_uni,theta_bi,theta_tri,prune_top_k,avg_acceptance,speedup' > "$RESULTS"

echo '[1/60] max_tree_size=4 theta=(0.8,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29600 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 4,0.8,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "4,0.8,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[2/60] max_tree_size=4 theta=(0.8,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29601 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 4,0.8,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "4,0.8,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[3/60] max_tree_size=4 theta=(0.8,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29602 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 4,0.8,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "4,0.8,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[4/60] max_tree_size=4 theta=(0.85,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29603 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 4,0.85,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "4,0.85,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[5/60] max_tree_size=4 theta=(0.85,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29604 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 4,0.85,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "4,0.85,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[6/60] max_tree_size=4 theta=(0.85,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29605 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 4,0.85,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "4,0.85,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[7/60] max_tree_size=4 theta=(0.9,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29606 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 4,0.9,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "4,0.9,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[8/60] max_tree_size=4 theta=(0.9,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29607 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 4,0.9,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "4,0.9,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[9/60] max_tree_size=4 theta=(0.9,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29608 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 4,0.9,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "4,0.9,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[10/60] max_tree_size=4 theta=(0.92,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29609 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 4,0.92,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "4,0.92,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[11/60] max_tree_size=4 theta=(0.92,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29610 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 4,0.92,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "4,0.92,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[12/60] max_tree_size=4 theta=(0.92,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29611 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 4 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 4,0.92,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "4,0.92,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[13/60] max_tree_size=6 theta=(0.8,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29612 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 6,0.8,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "6,0.8,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[14/60] max_tree_size=6 theta=(0.8,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29613 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 6,0.8,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "6,0.8,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[15/60] max_tree_size=6 theta=(0.8,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29614 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 6,0.8,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "6,0.8,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[16/60] max_tree_size=6 theta=(0.85,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29615 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 6,0.85,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "6,0.85,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[17/60] max_tree_size=6 theta=(0.85,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29616 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 6,0.85,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "6,0.85,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[18/60] max_tree_size=6 theta=(0.85,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29617 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 6,0.85,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "6,0.85,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[19/60] max_tree_size=6 theta=(0.9,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29618 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 6,0.9,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "6,0.9,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[20/60] max_tree_size=6 theta=(0.9,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29619 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 6,0.9,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "6,0.9,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[21/60] max_tree_size=6 theta=(0.9,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29620 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 6,0.9,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "6,0.9,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[22/60] max_tree_size=6 theta=(0.92,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29621 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 6,0.92,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "6,0.92,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[23/60] max_tree_size=6 theta=(0.92,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29622 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 6,0.92,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "6,0.92,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[24/60] max_tree_size=6 theta=(0.92,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29623 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 6 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 6,0.92,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "6,0.92,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[25/60] max_tree_size=8 theta=(0.8,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29624 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 8 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,0.8,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,0.8,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[26/60] max_tree_size=8 theta=(0.8,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29625 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 8 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,0.8,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,0.8,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[27/60] max_tree_size=8 theta=(0.8,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29626 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 8 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,0.8,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,0.8,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[28/60] max_tree_size=8 theta=(0.85,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29627 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 8 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,0.85,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,0.85,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[29/60] max_tree_size=8 theta=(0.85,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29628 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 8 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,0.85,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,0.85,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[30/60] max_tree_size=8 theta=(0.85,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29629 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 8 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,0.85,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,0.85,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[31/60] max_tree_size=8 theta=(0.9,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29630 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 8 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,0.9,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,0.9,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[32/60] max_tree_size=8 theta=(0.9,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29631 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 8 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,0.9,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,0.9,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[33/60] max_tree_size=8 theta=(0.9,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29632 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 8 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,0.9,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,0.9,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[34/60] max_tree_size=8 theta=(0.92,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29633 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 8 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,0.92,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,0.92,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[35/60] max_tree_size=8 theta=(0.92,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29634 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 8 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,0.92,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,0.92,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[36/60] max_tree_size=8 theta=(0.92,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29635 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 8 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 8,0.92,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "8,0.92,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[37/60] max_tree_size=10 theta=(0.8,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29636 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 10 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 10,0.8,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "10,0.8,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[38/60] max_tree_size=10 theta=(0.8,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29637 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 10 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 10,0.8,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "10,0.8,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[39/60] max_tree_size=10 theta=(0.8,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29638 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 10 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 10,0.8,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "10,0.8,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[40/60] max_tree_size=10 theta=(0.85,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29639 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 10 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 10,0.85,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "10,0.85,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[41/60] max_tree_size=10 theta=(0.85,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29640 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 10 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 10,0.85,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "10,0.85,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[42/60] max_tree_size=10 theta=(0.85,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29641 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 10 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 10,0.85,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "10,0.85,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[43/60] max_tree_size=10 theta=(0.9,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29642 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 10 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 10,0.9,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "10,0.9,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[44/60] max_tree_size=10 theta=(0.9,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29643 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 10 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 10,0.9,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "10,0.9,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[45/60] max_tree_size=10 theta=(0.9,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29644 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 10 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 10,0.9,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "10,0.9,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[46/60] max_tree_size=10 theta=(0.92,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29645 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 10 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 10,0.92,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "10,0.92,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[47/60] max_tree_size=10 theta=(0.92,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29646 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 10 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 10,0.92,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "10,0.92,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[48/60] max_tree_size=10 theta=(0.92,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29647 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 10 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 10,0.92,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "10,0.92,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[49/60] max_tree_size=12 theta=(0.8,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29648 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 12 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 12,0.8,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "12,0.8,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[50/60] max_tree_size=12 theta=(0.8,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29649 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 12 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 12,0.8,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "12,0.8,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[51/60] max_tree_size=12 theta=(0.8,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29650 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.8 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 12 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 12,0.8,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "12,0.8,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[52/60] max_tree_size=12 theta=(0.85,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29651 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 12 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 12,0.85,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "12,0.85,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[53/60] max_tree_size=12 theta=(0.85,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29652 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 12 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 12,0.85,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "12,0.85,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[54/60] max_tree_size=12 theta=(0.85,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29653 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.85 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 12 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 12,0.85,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "12,0.85,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[55/60] max_tree_size=12 theta=(0.9,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29654 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 12 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 12,0.9,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "12,0.9,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[56/60] max_tree_size=12 theta=(0.9,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29655 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 12 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 12,0.9,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "12,0.9,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[57/60] max_tree_size=12 theta=(0.9,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29656 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.9 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 12 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 12,0.9,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "12,0.9,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[58/60] max_tree_size=12 theta=(0.92,0.35,0.1) prune_top_k=4'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29657 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 12 --prune-top-k 4 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 12,0.92,0.35,0.1,4: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "12,0.92,0.35,0.1,4,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[59/60] max_tree_size=12 theta=(0.92,0.35,0.1) prune_top_k=6'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29658 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 12 --prune-top-k 6 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 12,0.92,0.35,0.1,6: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "12,0.92,0.35,0.1,6,$ACC,$SPEEDUP" >> "$RESULTS"

echo '[60/60] max_tree_size=12 theta=(0.92,0.35,0.1) prune_top_k=8'
OUT=$(PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --master_port=29659 benchmark.py --dataset mt-bench --max-samples 40 --max-new-tokens 512 --model-name-or-path Qwen/Qwen3-4B --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 --temperature 0.0 --dynamic-branching --theta-uni 0.92 --theta-bi 0.35 --theta-tri 0.1 --max-tree-size 12 --prune-top-k 8 2>&1)
SPEEDUP=$(echo "$OUT" | grep "Decoding speedup" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
ACC=$(echo "$OUT" | grep "Average Acceptance length" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
  echo "[WARN] Config 12,0.92,0.35,0.1,8: no results (benchmark may have failed)" >&2
  echo "$OUT" | tail -50 > logs/hp_search_debug_last.txt
fi
echo "12,0.92,0.35,0.1,8,$ACC,$SPEEDUP" >> "$RESULTS"
