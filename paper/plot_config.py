"""Shared display constants for Pareto and comparison figures."""

TREE_LABELS = {
    1: "v1 (threshold+cap)",
    2: "v2 (EAGLE-2)",
    3: "v3 (best-first)",
    4: "v4 (prefix-aware)",
}
TREE_LABELS_STR = {
    "v1": "Threshold (v1)",
    "v2": "EAGLE-2 (v2)",
    "v3": "Best-First (v3)",
    "v4": "Prefix-Aware (v4, ours)",
}
TREE_COLORS_INT = {1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", 4: "#984ea3"}
TREE_COLORS_STR = {"v1": "#d62728", "v2": "#1f77b4", "v3": "#2ca02c", "v4": "#9467bd"}
TREE_MARKERS_INT = {1: "o", 2: "s", 3: "^", 4: "D"}
TREE_MARKERS_STR = {"v1": "o", "v2": "s", "v3": "^", "v4": "D"}
BENCH_DISPLAY = {
    "mt-bench": "MT-Bench",
    "humaneval": "HumanEval",
    "math500": "MATH-500",
}
