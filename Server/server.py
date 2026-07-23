#!/usr/bin/env python3
"""
CLI entry point for the FL server.

Usage:
    python server.py --num_rounds 81 --budget 1000 --threshold 0.05
    python server.py --num_rounds 81 --budget 1000 --is_resume
"""
import argparse
import os
import json
import time
import sys
import numpy as np
import flwr as fl
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import FLstrategy as _flstrategy
import blockchain_service as _bcs  # also inserts repo root into sys.path
from blockchain_metrics import aggregate_run, aggregate_latency_comm_run
from FLstrategy import (
    SaveModelStrategy,
    get_evaluate_fn,
    get_on_fit_config_fn,
    evaluate_config,
    weighted_average,
)
from blockchain_service import BlockchainService
from flwr.common import parameters_to_ndarrays
from utils.model_dataset_registry import (
    build_model,
    get_dataset_input_shape,
    load_dataset_raw,
    normalize_dataset_name,
    write_model_architecture_json,
)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

blockchainService = BlockchainService()


def evaluate_final_model(model, run_dir, session_dir, seed, dataset, run_stamp, config_data):
    """Load the saved global model and evaluate it on the full held-out test set.

    Results are printed and written to a JSON file inside run_dir whose name
    follows the same pattern as the server log produced by run_seeds.sh:
        final_test_eval_{dataset}_{model_type}_seed{seed}_{num_clients}clients_
                         {client_selection}_alpha{alpha}_{run_stamp}.json
    """
    weights_path = os.path.abspath(
        f"../Server/model_weights/{session_dir}/Global_model_seed{seed}.npy"
    )
    if not os.path.exists(weights_path):
        print(f"[server] Final model not found at {weights_path} — skipping test evaluation.")
        return None

    # np.save on a 2-element tuple produces a shape-(2,) object array.
    # Index directly: [0] is the Parameters object, [1] is the metrics dict.
    loaded = np.load(weights_path, allow_pickle=True)
    parameters = loaded[0]
    ndarrays = parameters_to_ndarrays(parameters)
    model.set_weights(ndarrays)

    input_shape = get_dataset_input_shape(dataset)
    (_, _), (x_test, y_test) = load_dataset_raw(dataset)
    h, w, c = input_shape
    x_test = x_test.reshape(-1, h, w, c).astype("float32") / 255.0
    y_test = y_test.reshape(-1)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(
        f"[server] Final test-set evaluation — "
        f"loss={loss:.6f}  accuracy={accuracy:.6f}  "
        f"n_samples={len(y_test)}"
    )

    result = {
        "test_loss":        float(loss),
        "test_accuracy":    float(accuracy),
        "n_test_samples":   int(len(y_test)),
        "dataset":          dataset,
        "model_type":       config_data.get("model_type", "cnn"),
        "seed":             seed,
        "num_clients":      config_data.get("num_clients"),
        "client_selection": config_data.get("client_selection"),
        "alpha":            config_data.get("alpha"),
        "num_rounds":       config_data.get("num_rounds"),
        "run_stamp":        run_stamp,
    }
    out_name = (
        f"final_test_eval_{dataset}_{config_data.get('model_type', 'cnn')}"
        f"_seed{seed}_{config_data.get('num_clients')}clients"
        f"_{config_data.get('client_selection')}_alpha{config_data.get('alpha')}"
        f"_{run_stamp}.json"
    )
    out_path = os.path.join(run_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[server] Test evaluation saved to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--num_rounds",   type=int,   required=True,       help="Total number of FL rounds")
    parser.add_argument("--budget",       type=float, required=True,       help="Total reward budget")
    parser.add_argument("--clp",          type=int,   default=31,          help="First non-CLP round for client-sampling policy")
    parser.add_argument("--threshold",    type=float, default=0.05,        help="Relative-change threshold for dynamic CLP detection")
    parser.add_argument("--num_clients",          type=int, default=25,  help="Total number of FL clients")
    parser.add_argument("--min_available_clients", type=int, default=None, help="Minimum connected clients before training starts (default: num_clients)")
    parser.add_argument("--seed",         type=int,   default=345,        help="Random seed for NumPy, TensorFlow, and file naming")
    parser.add_argument("--dataset",      type=str,   default="cifar10",  help="Dataset name used in output file names (e.g. cifar10, fmnist)")
    parser.add_argument("--alpha",        type=float, default=0.1,        help="Dirichlet alpha for non-IID data splitting")
    parser.add_argument("--batch_size",   type=int,   default=32,         help="Local training batch size on clients")
    parser.add_argument("--local_epochs", type=int,   default=2,          help="Local training epochs per FL round on clients")
    parser.add_argument("--is_resume",    action="store_true", default=False, help="Resume from last session weights")
    parser.add_argument("--model_type",   type=str,   default="cnn",          help="Model architecture: cnn or alexnet")
    parser.add_argument("--ratio",        type=float, default=1.5,            help="Geometric growth ratio for per-client sample counts")
    parser.add_argument("--cfl_clients_per_round", type=int, default=3,       help="Number of clients sampled per round when using --cfl")
    sel_group = parser.add_mutually_exclusive_group()
    sel_group.add_argument("--clp_aware", action="store_true", help="CLP-aware sampling policy")
    sel_group.add_argument("--cfl",       action="store_true", help="Conventional blockchain-based FL sampling without CLP (default)")
    args = parser.parse_args()
    if args.min_available_clients is None:
        args.min_available_clients = args.num_clients

    # Resolve selection mode; default to cfl when neither flag is given
    client_selection = "clp_aware" if args.clp_aware else "cfl"

    dataset = normalize_dataset_name(args.dataset)
    if args.alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {args.alpha}")
    if args.clp <= 0:
        raise ValueError(f"clp must be > 0, got {args.clp}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {args.batch_size}")
    if args.local_epochs <= 0:
        raise ValueError(f"local_epochs must be > 0, got {args.local_epochs}")
    if args.cfl_clients_per_round <= 0:
        raise ValueError(f"cfl_clients_per_round must be > 0, got {args.cfl_clients_per_round}")

    # Apply seed and dataset to all modules that use them as module-level constants
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    _flstrategy.SEED    = args.seed
    _flstrategy.DATASET_NAME = dataset
    _bcs.SEED           = args.seed
    _bcs.DATASET_NAME   = dataset

    session = int(time.time())
    run_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(session))

    # Structured output directory: <dataset>_<model>_seed<seed>_<N>clients_<timestamp>
    run_dir = os.path.abspath(
        f"../report/temp/{dataset}_{args.model_type}_seed{args.seed}_{args.min_available_clients}clients_{client_selection}_alpha{args.alpha}_{run_stamp}"
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"[server] Output directory: {run_dir}")

    # FL session folder name: <dataset>_<model>_<N>clients_<selection>_seed<seed>
    session_dir = f"{dataset}_{args.min_available_clients}clients_{client_selection}_seed{args.seed}_{args.model_type}_{run_stamp}"
    model = build_model(dataset, args.model_type)
    write_model_architecture_json(model, "model_architecture.json")

    # Write config_training.json so FLstrategy.py and blockchain_service.py can read it
    config_data = {
        "num_rounds":   args.num_rounds,
        "is_resume":    args.is_resume,
        "session":      session,
        "run_stamp":    run_stamp,
        "session_dir":  session_dir,
        "clp":          args.clp,
        "threshold":    args.threshold,
        "alpha":        args.alpha,
        "batch_size":   args.batch_size,
        "local_epochs": args.local_epochs,
        "dataset":      dataset,
        "log_dir":          run_dir,
        "client_selection": client_selection,
        "model_type":       args.model_type,
        "num_clients":      args.min_available_clients,
        "ratio":            args.ratio,
        "cfl_clients_per_round": args.cfl_clients_per_round,
    }
    with open('config_training.json', 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"[server] Config written: {config_data}")

    # Create model_weights directory if needed
    if not os.path.exists('../Server/model_weights'):
        os.mkdir('../Server/model_weights')

    # Optionally resume from a previous session
    initial_params = None
    if args.is_resume:
        weights_path = f'../Server/model_weights/{session_dir}/Global_model_seed{args.seed}.npy'
        if os.path.exists(weights_path):
            initial_params = np.load(weights_path, allow_pickle=True)
            print(f"[server] Resuming from {weights_path}")
        else:
            print(f"[server] --is_resume set but no weights found at {weights_path}, starting fresh")

    frac_fit    = 1
    min_avail_c = args.min_available_clients
    min_fit_c   = min_avail_c
    min_eval_c  = max(1, min_avail_c // 2)

    strategy = SaveModelStrategy(
        fraction_fit=frac_fit,
        fraction_evaluate=0.6,
        min_fit_clients=min_fit_c,
        min_evaluate_clients=min_eval_c,
        min_available_clients=min_avail_c,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=get_on_fit_config_fn(),
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=initial_params,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        threshold=args.threshold,
        client_selection=client_selection,
        cfl_clients_per_round=args.cfl_clients_per_round,
    )

    blockchainService.addStrategy(session, 'FedAvg', args.num_rounds, min_avail_c)
    _bc_txs = blockchainService.metrics.transactions
    strategy.latency_comm_logger.log(
        part="blockchain", event="addStrategy", round_num=0,
        latency_s=_bc_txs[-1]["latency_s"] if _bc_txs else 0.0,
        comm_bytes=_bc_txs[-1].get("calldata_bytes", 0) if _bc_txs else 0,
    )

    print(f"[server] Starting Flower server on 127.0.0.1:8080 for {args.num_rounds} rounds …")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    print(f"[server] Evaluating final global model on held-out test set …")
    final_test_eval = evaluate_final_model(
        model, run_dir, session_dir, args.seed, dataset, run_stamp, config_data
    )
    if final_test_eval is not None:
        strategy.result["final_test_loss"] = final_test_eval["test_loss"]
        strategy.result["final_test_accuracy"] = final_test_eval["test_accuracy"]
        strategy.result["final_test_num_samples"] = final_test_eval["n_test_samples"]

    # Post-training: record contributions on blockchain and save results
    print(f"\n[server] Training done. Recording contributions …")
    blockchainService.snapshot_client_balances(strategy.contribution, "before_payment")
    for client in strategy.contribution.keys():
        if client == 'total_data_size':
            continue
        blockchainService.addContribution(
            _rNo=strategy.contribution[client]['num_rounds_participated'],
            _dataSize=strategy.contribution[client]['data_size'],
            _client_address=strategy.contribution[client]['client_address'],
            _totalDataSize=strategy.contribution['total_data_size'],
            _totalBudget=args.budget,
            number_of_rounds=args.num_rounds,
            client_id=client,
            joined_round=strategy.contribution[client]['joined_round'],
        )
        # addContribution is now split into two TXs:
        #   transactions[-2] = calculateContribution  (state update, has calldata_bytes)
        #   transactions[-1] = paymentTransfer         (ETH send, has payment/balance info)
        bc_txs = blockchainService.metrics.transactions
        strategy.latency_comm_logger.log(
            part="blockchain", event="addContribution", round_num=0,
            latency_s=bc_txs[-2]["latency_s"] if len(bc_txs) >= 2 else 0.0,
            comm_bytes=bc_txs[-2].get("calldata_bytes", 0) if len(bc_txs) >= 2 else 0,
            client_id=client,
        )
        strategy.latency_comm_logger.log(
            part="blockchain", event="paymentTransfer", round_num=0,
            latency_s=bc_txs[-1]["latency_s"] if bc_txs else 0.0,
            comm_bytes=0,
            client_id=client,
        )

    blockchainService.snapshot_client_balances(strategy.contribution, "after_payment")

    with open(os.path.join(run_dir, f'cfl_acc_loss_{dataset}_{args.model_type}_{args.min_available_clients}clients_{client_selection}_seed{args.seed}_{run_stamp}.json'), 'w') as f:
        json.dump(strategy.result, f, indent=2)
    with open(os.path.join(run_dir, f'cfl_contribution_profile_{dataset}_{args.model_type}_{args.min_available_clients}clients_{client_selection}_seed{args.seed}_{run_stamp}.json'), 'w') as f:
        json.dump(strategy.contribution, f, indent=2)

    blockchainService.metrics.save(run_dir, run_tag=run_stamp)
    strategy.latency_comm_logger.save(run_dir, run_stamp)
    aggregate_run(run_dir, run_tag=run_stamp)
    aggregate_latency_comm_run(run_dir, run_tag=run_stamp)
    print(f"[server] Results saved to {run_dir}")


if __name__ == "__main__":
    main()
