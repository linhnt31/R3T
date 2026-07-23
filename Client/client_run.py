#!/usr/bin/env python3
"""
CLI entry point for an FL client.

Usage:
    python client_run.py --client_id 0
    python client_run.py --client_id 3 --server_address 127.0.0.1:8080
"""
import argparse
import csv
import json
import os
import time
import sys
import numpy as np
import tensorflow as tf
import flwr as fl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client import CifarClient
import blockchain_service as _bcs
from blockchain_service import BlockchainService
from blockchain_metrics import LatencyCommLogger
from utils.model_dataset_registry import (
    build_model,
    get_dataset_input_shape,
    load_dataset_raw,
    normalize_dataset_name,
    write_model_architecture_json,
)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def configure_device():
    """Auto-detect GPU(s); enable memory growth so multiple clients can share the GPU.
    Falls back to CPU silently if no GPU is found."""
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[device] GPU(s) detected: {[g.name for g in gpus]} — memory growth enabled")
        except RuntimeError as e:
            print(f"[device] GPU config warning: {e}")
    else:
        print(f"[device] No GPU found — running on CPU: {[c.name for c in cpus]}")

blockchainService = BlockchainService()


def _generate_sample_counts(num_clients: int, total_samples: int, ratio: float = 1.5) -> np.ndarray:
    """
    Generate num_clients strictly-increasing integer sample counts that sum exactly to
    total_samples.  Client 0 receives the fewest samples; client num_clients-1 the most.

    ratio controls how fast counts grow: each client gets ~ratio× more data than the
    previous one (default 1.5 → +50% per step).
    """
    weights = np.array([ratio ** i for i in range(num_clients)], dtype=float)
    weights /= weights.sum()

    # Floor allocation + largest-remainder to hit the exact total
    raw = weights * total_samples
    counts = np.floor(raw).astype(int)
    remainder = total_samples - int(counts.sum())
    order = np.argsort(raw - counts)[::-1]
    counts[order[:remainder]] += 1

    # Ensure every client gets at least 8 samples before enforcing monotonicity.
    counts[0] = max(counts[0], 8)

    # Enforce strict monotonicity (safety net for extreme ratios / small totals)
    for i in range(1, num_clients):
        if counts[i] <= counts[i - 1]:
            counts[i] = counts[i - 1] + 1

    # Re-absorb any sum drift caused by the monotonicity pass into the last bucket
    counts[-1] += total_samples - int(counts.sum())

    assert int(counts.sum()) == total_samples, \
        f"_generate_sample_counts: sum {counts.sum()} != {total_samples}"
    assert all(counts[i] < counts[i + 1] for i in range(num_clients - 1)), \
        "_generate_sample_counts: result is not strictly increasing"

    return counts


def load_dataset(client_id: int, dataset: str = "cifar10", alpha: float = 0.1,
                 num_clients: int = 15, ratio: float = 1.5, split_log_path: str = None):
    dataset = normalize_dataset_name(dataset)
    input_shape = get_dataset_input_shape(dataset)
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    assert client_id in range(num_clients)

    (x_train, y_train), (x_test, y_test) = load_dataset_raw(dataset)
    h, w, c = input_shape
    x_train = x_train.reshape(-1, h, w, c).astype("float32") / 255.0
    x_test  = x_test.reshape(-1, h, w, c).astype("float32")  / 255.0

    test_random_indices = np.random.choice(x_test.shape[0], 5000, replace=False)

    samples_per_client = _generate_sample_counts(num_clients, len(y_train), ratio)

    def _largest_remainder(total: int, probs: np.ndarray) -> np.ndarray:
        """Convert probabilities into integer counts that sum exactly to total."""
        raw = probs * total
        counts = np.floor(raw).astype(int)
        remainder = int(total - counts.sum())
        if remainder > 0:
            order = np.argsort(raw - counts)[::-1]
            counts[order[:remainder]] += 1
        return counts

    def _allocate_with_caps(total: int, probs: np.ndarray, caps: np.ndarray) -> np.ndarray:
        """
        Allocate 'total' items by probs, but never exceed caps.
        Guarantees exact allocation when sum(caps) >= total.
        """
        if total == 0:
            return np.zeros_like(caps, dtype=int)
        if int(caps.sum()) < total:
            raise ValueError("Insufficient capacity to allocate class samples.")

        alloc = np.minimum(_largest_remainder(total, probs), caps)
        while int(alloc.sum()) < total:
            remaining = total - int(alloc.sum())
            slack_idx = np.where(caps > alloc)[0]
            if len(slack_idx) == 0:
                raise ValueError("No slack left while allocation is incomplete.")
            slack_probs = probs[slack_idx]
            if float(slack_probs.sum()) == 0.0:
                slack_probs = np.full(len(slack_idx), 1.0 / len(slack_idx))
            else:
                slack_probs = slack_probs / slack_probs.sum()

            add = _largest_remainder(remaining, slack_probs)
            add = np.minimum(add, caps[slack_idx] - alloc[slack_idx])
            alloc[slack_idx] += add
        return alloc

    y_flat = y_train.reshape(-1)
    class_indices = [np.where(y_flat == class_id)[0] for class_id in range(10)]
    for idx in class_indices:
        np.random.shuffle(idx)

    client_indices = [[] for _ in range(num_clients)]
    remaining_capacity = samples_per_client.astype(int).copy()
    class_ptr = np.zeros(10, dtype=int)

    # Ensure each client sees each class at least once (when feasible).
    for class_id in range(10):
        for client_idx in range(num_clients):
            if remaining_capacity[client_idx] <= 0:
                continue
            ptr = class_ptr[class_id]
            if ptr >= len(class_indices[class_id]):
                break
            sample_idx = class_indices[class_id][ptr]
            client_indices[client_idx].append(sample_idx)
            class_ptr[class_id] += 1
            remaining_capacity[client_idx] -= 1

    # Allocate the rest class-by-class with Dirichlet heterogeneity and hard capacity caps.
    for class_id in range(10):
        ptr = class_ptr[class_id]
        pool = class_indices[class_id][ptr:]
        n_pool = len(pool)
        if n_pool == 0:
            continue

        active = np.where(remaining_capacity > 0)[0]
        probs = np.random.dirichlet(np.full(len(active), alpha))
        alloc_active = _allocate_with_caps(n_pool, probs, remaining_capacity[active])

        start = 0
        for local_idx, client_idx in enumerate(active):
            count = int(alloc_active[local_idx])
            if count == 0:
                continue
            end = start + count
            client_indices[client_idx].extend(pool[start:end].tolist())
            remaining_capacity[client_idx] -= count
            start = end

    if int(remaining_capacity.sum()) != 0:
        raise ValueError(f"Split failed to satisfy all per-client quotas: {remaining_capacity.tolist()}")

    # Extract only this client's slice — avoid materialising all N clients' arrays
    my_indices = np.array(client_indices[client_id])
    x_tr = x_train[my_indices]
    y_tr = y_flat[my_indices].reshape(-1)

    if split_log_path is not None:
        split_dir = os.path.dirname(split_log_path)
        if split_dir:
            os.makedirs(split_dir, exist_ok=True)

        fieldnames = [
            "dataset",
            "alpha",
            "client_id",
            "num_samples",
            "num_classes_present",
            "classes_present",
        ] + [f"class_{i}" for i in range(10)]
        with open(split_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Iterate over index lists — no need to hold all x slices in RAM
            for idx, idxs in enumerate(client_indices):
                y_split = y_flat[np.array(idxs)]
                class_counts = np.bincount(y_split.astype(int), minlength=10)
                classes_present = [int(c) for c in np.where(class_counts > 0)[0]]
                row = {
                    "dataset": dataset,
                    "alpha": alpha,
                    "client_id": idx,
                    "num_samples": int(len(y_split)),
                    "num_classes_present": len(classes_present),
                    "classes_present": ",".join(map(str, classes_present)),
                }
                for class_id, count in enumerate(class_counts):
                    row[f"class_{class_id}"] = int(count)
                writer.writerow(row)
        print(f"[client {client_id}] data split summary CSV saved to {split_log_path}")

    # Free the full training array — only x_tr (this client's slice) is needed going forward
    del x_train, y_flat, client_indices

    return (x_tr, y_tr), (x_test[test_random_indices], y_test[test_random_indices].reshape(-1))


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--client_id",      type=int, required=True,            help="Client ID (0-9)")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080", help="FL server address")
    parser.add_argument("--seed",            type=int, default=345,             help="Random seed for NumPy and TensorFlow")
    parser.add_argument("--dataset",         type=str, default=None,            help="Dataset: cifar10 or fmnist. If omitted, read from server config.")
    parser.add_argument("--alpha",           type=float, default=None,          help="Dirichlet alpha for non-IID split. If omitted, read from server config.")
    parser.add_argument("--log_dir",         type=str, default=None,            help="Output directory for metrics (defaults to run dir from server config)")
    parser.add_argument("--model_type",      type=str,   default=None,            help="Model architecture: cnn or alexnet. If omitted, read from server config.")
    parser.add_argument("--num_clients",     type=int,   default=None,            help="Total number of clients (controls data split). If omitted, read from server config.")
    parser.add_argument("--ratio",           type=float, default=None,            help="Geometric growth ratio for sample counts (default 1.5). If omitted, read from server config.")
    args = parser.parse_args()

    server_cfg = {}
    if os.path.exists('../Server/config_training.json'):
        with open('../Server/config_training.json', 'r') as f:
            server_cfg = json.load(f)

    dataset = normalize_dataset_name(args.dataset or server_cfg.get("dataset", "cifar10"))
    alpha = float(args.alpha if args.alpha is not None else server_cfg.get("alpha", 0.1))
    model_type = args.model_type or server_cfg.get("model_type", "cnn")
    num_clients = int(args.num_clients if args.num_clients is not None else server_cfg.get("num_clients", 15))
    ratio = float(args.ratio if args.ratio is not None else server_cfg.get("ratio", 1.5))
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    configure_device()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    _bcs.SEED         = args.seed
    _bcs.DATASET_NAME = dataset

    # Reinitialise with the correct client_id so the metrics role label is "client_N"
    global blockchainService
    blockchainService = BlockchainService(client_id=args.client_id)
    client_address = blockchainService.getAddress(args.client_id)
    lc_logger = LatencyCommLogger(role=f"client_{args.client_id}")
    print(f"[client {args.client_id}] address={client_address}")

    # Resolve log directory early so we can write split logs there before training starts.
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = server_cfg.get('log_dir', '../report/temp')
    os.makedirs(log_dir, exist_ok=True)
    run_stamp = server_cfg.get("run_stamp", time.strftime("%Y%m%d_%H%M%S"))

    split_log_path = os.path.join(log_dir, f"data_split_{dataset}_seed{args.seed}_{run_stamp}.csv")

    model = build_model(dataset, model_type, learning_rate=3e-4)
    write_model_architecture_json(model, "model_architecture.json")
    (x_train, y_train), (x_test, y_test) = load_dataset(
        args.client_id, dataset=dataset, alpha=alpha,
        num_clients=num_clients, ratio=ratio, split_log_path=split_log_path
    )
    print(f"[client {args.client_id}] train={len(x_train)}  test={len(x_test)}")

    print(f"[client {args.client_id}] Connecting to {args.server_address} …")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=CifarClient(model, x_train, y_train, x_test, y_test,
                           args.client_id, client_address,
                           blockchain_service=blockchainService,
                           latency_comm_logger=lc_logger),
        grpc_max_message_length=1024 * 1024 * 1024,
    )

    # Save client metrics into the same run directory.
    blockchainService.metrics.save(log_dir, run_tag=run_stamp)
    lc_logger.save(log_dir, run_tag=run_stamp)


if __name__ == "__main__":
    main()
