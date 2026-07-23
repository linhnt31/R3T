"""
Compute mean metrics across 4 seeds (8, 42, 345, 1234) for each
(method × dataset × alpha) combination and export to CSV and XLSX.

Metrics extracted:
  1. cfl_acc_loss*.json          -> aggregated_accuracy (final round)
  2. latency_comm_server*summary.json
       blockchain.<key>.total_latency_s / total_comm_bytes  (all keys)
       training.<key>.total_latency_s  / total_comm_bytes   (all keys)
  3. blockchain_metrics_overall*.json
       observation_span_s   (top-level)
       gas.total            (top-level)
       per_function.<func>.total_gas        (per function)
       per_function.<func>.throughput_tps   (per function)
"""

import os
import re
import json
import glob
from collections import defaultdict

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE)
RESULTS_DIR = os.path.join(PROJECT_DIR, "name_of_your_results_dir")  # <-- UPDATE THIS

METHODS  = ["cfl", "proposed"]
DATASETS = ["cifar10", "fmnist"]
ALPHAS   = {"alpha01": 0.1, "alpha04": 0.4, "alpha07": 0.7}
# SEEDS    = [1234, 20, 7, 31, 41, 42, 100]
SEEDS    = [1234, 41, 42, 100, 7, 20]

# ── Helpers ────────────────────────────────────────────────────────────────────

# Matches: [server] Final test-set evaluation — loss=X  accuracy=Y  n_samples=Z
_EVAL_RE = re.compile(r"Final test-set evaluation.*?accuracy=([0-9.]+(?:e[+-]?\d+)?)")


def _open_path(p):
    """Return a path string safe for open() on Windows (handles MAX_PATH > 260)."""
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(p)
    return p


def extract_log_accuracy(log_path):
    """Return the final test-set accuracy from a server*.log file, or None."""
    with open(_open_path(log_path), encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _EVAL_RE.search(line)
            if m:
                return float(m.group(1))
    return None


def find_run_dir(method, dataset, alpha_key, seed):
    """Return the single run directory for the given (method, dataset, alpha, seed).

    Searches every batch directory under RESULTS_DIR, e.g.
      2026_Rebuttal_results/20clients_20260415/{method}/{dataset}/{alpha_key}/*seed{seed}*/
    """
    matches = []
    for batch_dir in glob.glob(os.path.join(RESULTS_DIR, "10clients*")):
        if not os.path.isdir(batch_dir):
            continue
        pattern = os.path.join(batch_dir, method, dataset, alpha_key, f"*seed{seed}*")
        matches += [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  WARNING: multiple dirs matched for seed{seed} in "
              f"{method}/{dataset}/{alpha_key} — using first: {matches[0]}")
    return matches[0]


def load_json(path):
    with open(_open_path(path), encoding="utf-8") as f:
        return json.load(f)


def find_file(run_dir, glob_pattern):
    """Return first file matching glob_pattern inside run_dir, or None."""
    matches = glob.glob(os.path.join(run_dir, glob_pattern))
    return matches[0] if matches else None


# ── Extractors ─────────────────────────────────────────────────────────────────

def extract_accuracy(run_dir):
    """All rounds of aggregated_accuracy from cfl_acc_loss*.json as {int_round: value}."""
    fp = find_file(run_dir, "cfl_acc_loss*.json")
    if not fp:
        return {}
    data = load_json(fp)
    acc = data.get("aggregated_accuracy", {})
    # Return dict keyed by integer round number
    return {int(k): v for k, v in acc.items() if v is not None}


def extract_latency_comm(run_dir):
    """total_latency_s and total_comm_bytes for all keys in blockchain and training."""
    fp = find_file(run_dir, "latency_comm_server*summary*.json")
    if not fp:
        fp = find_file(run_dir, "latency_comm_server*.json")
    if not fp:
        return {}

    data = load_json(fp)
    parts = data.get("parts", {})
    result = {}

    for section in ("blockchain", "training"):
        section_data = parts.get(section, {})
        for key, val in section_data.items():
            prefix = f"{section}.{key}"
            result[f"{prefix}.total_latency_s"]  = val.get("total_latency_s")
            result[f"{prefix}.total_comm_bytes"]  = val.get("total_comm_bytes")

    return result


def extract_blockchain_metrics(run_dir):
    """observation_span_s, gas.total and per_function total_gas / throughput_tps."""
    fp = find_file(run_dir, "blockchain_metrics_overall*.json")
    if not fp:
        return {}

    data = load_json(fp)
    result = {
        "observation_span_s":        data.get("observation_span_s"),
        "throughput_tps_processing": data.get("throughput_tps_processing"),
        "gas.total":                 data.get("gas", {}).get("total"),
    }

    for func, val in data.get("per_function", {}).items():
        result[f"per_function.{func}.total_gas"]      = val.get("total_gas")
        result[f"per_function.{func}.throughput_tps"] = val.get("throughput_tps")

    return result


# ── Main loop ──────────────────────────────────────────────────────────────────

rows = []
# accuracy_per_round[combo_label][round] = list of values across seeds
accuracy_per_round: dict[str, dict[int, list]] = {}

for method in METHODS:
    for dataset in DATASETS:
        for alpha_key, alpha_val in ALPHAS.items():
            print(f"\n{'='*60}")
            print(f"  {method} / {dataset} / alpha={alpha_val}")

            combo_label = f"{method}|{dataset}|alpha{alpha_val}"

            # Accumulate values per metric across seeds
            seed_buckets: dict[str, list] = defaultdict(list)
            round_buckets: dict[int, list] = defaultdict(list)
            seeds_found = []

            for seed in SEEDS:
                run_dir = find_run_dir(method, dataset, alpha_key, seed)
                if not run_dir:
                    print(f"  [MISSING] seed={seed}")
                    continue

                print(f"  [OK] seed={seed}  ->  {os.path.basename(run_dir)}")
                seeds_found.append(seed)

                # Per-round accuracy
                acc_rounds = extract_accuracy(run_dir)
                for rnd, val in acc_rounds.items():
                    round_buckets[rnd].append(val)

                # Other metrics (latency, comm, blockchain)
                metrics = {}
                metrics.update(extract_latency_comm(run_dir))
                metrics.update(extract_blockchain_metrics(run_dir))

                for k, v in metrics.items():
                    seed_buckets[k].append(v)

            # Store mean accuracy per round for this combo
            accuracy_per_round[combo_label] = {
                rnd: sum(vals) / len(vals)
                for rnd, vals in round_buckets.items()
                if vals
            }

            # Build one summary row (other metrics only)
            row = {
                "method":      method,
                "dataset":     dataset,
                "alpha":       alpha_val,
                "seeds_used":  ",".join(str(s) for s in seeds_found),
                "n_seeds":     len(seeds_found),
            }
            for metric, values in seed_buckets.items():
                valid = [v for v in values if v is not None]
                row[metric] = sum(valid) / len(valid) if valid else None

            rows.append(row)

# ── Scan server*.log files for final test-set accuracy ────────────────────────
# log_acc[(method, dataset, alpha_val)][seed] = accuracy

log_acc: dict[tuple, dict[int, float | None]] = defaultdict(dict)
_batch_dirs: list[str] = []

for batch_dir in glob.glob(os.path.join(RESULTS_DIR, "10clients*")):
    if not os.path.isdir(batch_dir):
        continue
    _batch_dirs.append(batch_dir)
    for method in METHODS:
        for dataset in DATASETS:
            for alpha_key, alpha_val in ALPHAS.items():
                folder = os.path.join(batch_dir, method, dataset, alpha_key)
                if not os.path.isdir(folder):
                    continue
                for log_path in glob.glob(os.path.join(folder, "server*.log")):
                    fname = os.path.basename(log_path)
                    m = re.search(r"seed(\d+)", fname)
                    if not m:
                        continue
                    seed = int(m.group(1))
                    if seed not in SEEDS:
                        continue
                    log_acc[(method, dataset, alpha_val)][seed] = extract_log_accuracy(log_path)

# Build log accuracy DataFrame (rows = combos, cols = per-seed + mean)
_all_log_seeds = sorted({s for d in log_acc.values() for s in d})
log_acc_rows = []
for (method, dataset, alpha_val), seed_dict in sorted(log_acc.items()):
    row = {"method": method, "dataset": dataset, "alpha": alpha_val}
    valid = []
    for seed in _all_log_seeds:
        acc = seed_dict.get(seed)
        row[f"seed_{seed}"] = acc
        if acc is not None:
            valid.append(acc)
    row["mean_accuracy"] = sum(valid) / len(valid) if valid else None
    row["n_seeds"] = len(valid)
    log_acc_rows.append(row)

_log_seed_cols = [f"seed_{s}" for s in _all_log_seeds]
log_acc_df = pd.DataFrame(log_acc_rows)[
    ["method", "dataset", "alpha"] + _log_seed_cols + ["mean_accuracy", "n_seeds"]
]

# ── Build main DataFrame (latency / comm / blockchain metrics) ─────────────────

df = pd.DataFrame(rows)

id_cols     = ["method", "dataset", "alpha", "seeds_used", "n_seeds"]
metric_cols = [c for c in df.columns if c not in id_cols]
df = df[id_cols + metric_cols]

# ── Build per-round accuracy DataFrame ────────────────────────────────────────
# Rows = round numbers, columns = one per (method, dataset, alpha) combo

all_rounds = sorted({rnd for combo in accuracy_per_round.values() for rnd in combo})
acc_df = pd.DataFrame(index=all_rounds)
acc_df.index.name = "round"

for combo_label, round_means in accuracy_per_round.items():
    acc_df[combo_label] = pd.Series(round_means)

acc_df = acc_df.sort_index()

# ── Export ─────────────────────────────────────────────────────────────────────

seeds_tag = f"{len(SEEDS)}seeds_" + "_".join(str(s) for s in SEEDS)
# Save next to proposed/ and cfl_worstcase/ (inside each batch dir).
# Fall back to RESULTS_DIR when there are multiple batch directories.
_out_dir = _batch_dirs[0] if len(_batch_dirs) == 1 else RESULTS_DIR
out_csv  = os.path.join(_out_dir, f"results_mean_across_{seeds_tag}.csv")
out_xlsx = os.path.join(_out_dir, f"results_mean_across_{seeds_tag}.xlsx")

df.to_csv(out_csv, index=False)
print(f"\nSaved CSV  -> {out_csv}")


def autofit(ws):
    for col_cells in ws.columns:
        max_len = max(
            len(str(cell.value)) if cell.value is not None else 0
            for cell in col_cells
        )
        ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 50)


with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    # Sheet 1: all metrics except accuracy
    df.to_excel(writer, index=False, sheet_name="means")
    autofit(writer.sheets["means"])

    # Sheet 2: mean aggregated_accuracy per round
    acc_df.to_excel(writer, sheet_name="accuracy_per_round")
    autofit(writer.sheets["accuracy_per_round"])

    # Sheet 3: final test-set accuracy per seed from server*.log
    log_acc_df.to_excel(writer, index=False, sheet_name="log_final_accuracy")
    autofit(writer.sheets["log_final_accuracy"])

print(f"Saved XLSX -> {out_xlsx}")
print(f"  Sheet 'blockchain_training_overhead'             : {len(df)} rows x {len(df.columns)} cols")
print(f"  Sheet 'accuracy_per_round': {len(acc_df)} rounds x {len(acc_df.columns)} combos")
print(f"  Sheet 'log_final_accuracy': {len(log_acc_df)} rows x {len(log_acc_df.columns)} cols")

# ── Quick preview ──────────────────────────────────────────────────────────────

print("\n-- Accuracy per round (first 5 rounds, all combos) --")
print(acc_df.head().to_string())
