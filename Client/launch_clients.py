#!/usr/bin/env python3
"""
Launch multiple FL clients in parallel from a single terminal.

Usage:
    python launch_clients.py --num_clients 10
    python launch_clients.py --num_clients 20 --server_address 127.0.0.1:8080
    python launch_clients.py --client_ids 0 1 3 7   # specific subset
"""
import argparse
import subprocess
import sys
import os
import time
import signal
import json

def main():
    parser = argparse.ArgumentParser(description="Launch FL clients in parallel")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--num_clients",  type=int,       help="Launch clients 0 … N-1")
    group.add_argument("--client_ids",   type=int, nargs="+", help="Launch specific client IDs")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--seed",    type=int, default=345,       help="Random seed forwarded to each client")
    parser.add_argument("--dataset", type=str, default=None,      help="Optional dataset override forwarded to each client")
    parser.add_argument("--alpha",      type=float, default=None,    help="Optional non-IID alpha override forwarded to each client")
    parser.add_argument("--model_type", type=str,   default=None,    help="Optional model type override forwarded to each client (cnn or alexnet)")
    parser.add_argument("--delay",   type=float, default=2.0,
                        help="Seconds between launching each client (avoids dataset-load collisions)")
    parser.add_argument("--device",  type=str,   default="auto",
                        help="Device for TensorFlow: 'auto' (GPU if available), 'cpu' (force CPU), "
                             "or 'gpu:N' / 'N' to pin to a specific GPU index (default: auto)")
    args = parser.parse_args()

    client_ids = list(range(args.num_clients)) if args.num_clients else args.client_ids
    n_clients  = len(client_ids)

    server_cfg = {}
    if os.path.exists("../Server/config_training.json"):
        try:
            with open("../Server/config_training.json", "r") as f:
                server_cfg = json.load(f)
        except json.JSONDecodeError:
            server_cfg = {}

    run_stamp = server_cfg.get("run_stamp", time.strftime("%Y%m%d_%H%M%S"))
    # Prefer server-provided run directory so clients/logs stay in the same results folder.
    if "log_dir" in server_cfg:
        run_dir = os.path.abspath(server_cfg["log_dir"])
    else:
        run_dataset = args.dataset if args.dataset is not None else server_cfg.get("dataset", "cifar10")
        run_alpha = server_cfg.get("alpha", args.alpha if args.alpha is not None else 0.1)
        run_selection = server_cfg.get("client_selection", "cfl")
        run_dir = os.path.abspath(
            f"../report/temp/{run_dataset}_seed{args.seed}_{n_clients}clients_{run_selection}_alpha{run_alpha}_{run_stamp}"
        )
    log_dir = os.path.join(run_dir, "client_logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[launcher] Output directory: {run_dir}")

    # Resolve CUDA_VISIBLE_DEVICES from --device flag
    device = args.device.strip().lower()
    if device == "cpu":
        cuda_visible = "-1"
    elif device == "auto":
        cuda_visible = None  # inherit from environment; TF picks GPU automatically
    else:
        # Accept "gpu:0", "gpu:1", "0", "1", etc.
        gpu_idx = device.replace("gpu:", "").replace("gpu", "0")
        cuda_visible = gpu_idx
    if cuda_visible is not None:
        label = "CPU only" if cuda_visible == "-1" else f"GPU {cuda_visible}"
        print(f"[launcher] Device: {label}  (CUDA_VISIBLE_DEVICES={cuda_visible})")
    else:
        print("[launcher] Device: auto (GPU if available, else CPU)")

    client_env = os.environ.copy()
    if cuda_visible is not None:
        client_env["CUDA_VISIBLE_DEVICES"] = cuda_visible

    processes = []
    print(f"[launcher] Launching {n_clients} client(s): {client_ids}")

    for cid in client_ids:
        log_path = os.path.join(log_dir, f"client_{cid}_{run_stamp}.log")
        log_file = open(log_path, "w")
        cmd = [
            sys.executable, "client_run.py",
            "--client_id", str(cid),
            "--server_address", args.server_address,
            "--seed", str(args.seed),
            "--log_dir", run_dir,
        ]
        # Only pass dataset/alpha when user explicitly overrides.
        # Otherwise, client_run.py auto-reads them from Server/config_training.json.
        if args.dataset is not None:
            cmd.extend(["--dataset", args.dataset])
        if args.alpha is not None:
            cmd.extend(["--alpha", str(args.alpha)])
        if args.model_type is not None:
            cmd.extend(["--model_type", args.model_type])

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            env=client_env,
        )
        processes.append((cid, proc, log_file))
        print(f"  [client {cid}] pid={proc.pid}  log={log_path}")
        time.sleep(args.delay)

    print(f"\n[launcher] All {len(processes)} client(s) started. Waiting for them to finish …")
    print("[launcher] Press Ctrl+C to stop all clients.\n")

    def shutdown(signum, frame):
        print("\n[launcher] Interrupted — terminating all clients …")
        for cid, proc, log_file in processes:
            proc.terminate()
            log_file.close()
            print(f"  [client {cid}] terminated")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait for all clients and report exit codes
    for cid, proc, log_file in processes:
        proc.wait()
        log_file.close()
        status = "OK" if proc.returncode == 0 else f"exit {proc.returncode}"
        print(f"  [client {cid}] finished — {status}  (log: {log_dir}/client_{cid}_{run_stamp}.log)")

    print("\n[launcher] All clients done.")


if __name__ == "__main__":
    main()
