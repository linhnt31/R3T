import os
import time
import numpy as np
import hashlib
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json

import flwr as fl
from flwr.common import Metrics, Parameters, Scalar
import tensorflow as tf
from blockchain_service import *
from blockchain_metrics import LatencyCommLogger
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_dataset_registry import get_dataset_input_shape, load_dataset_raw, normalize_dataset_name

SEED = 345  # overridden at runtime by server.py via FLstrategy.SEED = args.seed
DATASET_NAME = "cifar10"  # overridden at runtime by server.py via FLstrategy.DATASET_NAME = args.dataset
    
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)

from pinatapy import PinataPy

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

with open('../api_key.json', 'r') as api:
    keys=api.read()
    data = json.loads(keys)
    api_key=data['api_key']
    secret_key=data['secret_key']
pinata = PinataPy(api_key, secret_key)

### Instantiate the blockchain service
blockchainService = BlockchainService()


'''
Reference: https://flower.ai/docs/framework/explanation-federated-evaluation.html#configuring-federated-evaluation
'''
class SaveModelStrategy(fl.server.strategy.FedAvg):
    ''' Note for choosing arguments in Strategy class
    Ref: https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html#Starting-the-training
    '''
    def __init__(self,# child's class
        *,
        fraction_fit=1, # Sample 30% of available clients for training
        fraction_evaluate=0.5, # Sample 50% of available clients for evaluation
        min_fit_clients=2, # Never sample less than 2 clients for training
        min_evaluate_clients=1, # Never sample less than 1 clients for evaluation
        min_available_clients=10, # Wait until all 10 clients are available
        evaluate_fn=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        on_fit_config_fn=None, #
        on_evaluate_config_fn=None,
        initial_parameters:fl.common.Parameters = None,
        threshold: float = 0.05,  # Relative change threshold for CLP detection
        client_selection: str = "cfl",  # "clp_aware" or "cfl"
        cfl_clients_per_round: int = 3,  # Used only when client_selection == "cfl"
        ) -> None:
        super().__init__( # parent'class
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        # Logger for per-round training & blockchain latency/communication
        self.latency_comm_logger = LatencyCommLogger(role="server")

        '''This value will be called in *server_api* to store the contribution of each client'''
        self.contribution={
            'total_data_size': 0
        }
        self.result={
            'aggregated_loss':{
                0:0
            },
            'aggregated_accuracy':{
                0:0
            },
            'fgn': {
                0: 0.0
            },
            'fgn_relative_change': {},   # (fgn(t) - fgn(t-1)) / fgn(t-1) per round
            'joined_client_ids': {}      # actual client_ids that returned fit results each round
        }

        # CLP detection state
        self.threshold = threshold      # Threshold for (fgn(t) - fgn(t-1)) / fgn(t-1)
        self.client_selection = client_selection
        self.cfl_clients_per_round = max(1, int(cfl_clients_per_round))
        # Reuse self.result["fgn"] as the authoritative store to avoid duplicate
        # per-round state in memory while preserving behavior.
        self.fgn: Dict[int, float] = self.result["fgn"]
        self.is_clp: Dict[int, bool] = {}       # CLP detection result per round
        self.client_ids_by_cid: Dict[str, int] = {}
        # Automatically discovered first non-CLP round.
        # Set in aggregate_fit once relative_change drops below threshold.
        self.non_clp_start_round: Optional[int] = None
        # Per-round wall-clock start time (set in configure_fit, read in aggregate_fit)
        # Captures the full round latency: client training + gRPC + aggregation
        self._round_start: Dict[int, float] = {}
        # Per-round count of selected clients (for server upload comm accounting)
        self._round_selected_count: Dict[int, int] = {}
        
    '''
    Reference: https://flower.ai/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html#Build-a-Strategy-from-scratch
    '''
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        available = list(client_manager.all().values())

        # Bootstrap client IDs for any proxy not yet seen in aggregate_fit
        for proxy in available:
            if proxy.cid not in self.client_ids_by_cid:
                try:
                    props_res = proxy.get_properties(GetPropertiesIns(config={}), timeout=10)
                    cid_val = props_res.properties.get("client_id")
                    if cid_val is not None:
                        self.client_ids_by_cid[proxy.cid] = int(cid_val)
                except Exception:
                    pass  # ID will be learned after first fit round

        available_count = len(available)
        _, min_num_clients = self.num_fit_clients(available_count)
        if available_count < min_num_clients:
            print(
                f"[configure_fit] round={server_round}, waiting: "
                f"available={available_count} < min_required={min_num_clients}"
            )
            return []

        if self.client_selection == "clp_aware":
            # `clp` marks the first non-CLP round for this sampling policy.
            # Keep it as fallback if dynamic detection has not triggered yet.
            non_clp_start_round = 31
            try: # Updated CLP value based on relative change threshold in config_training.json if available; otherwise use default of 31.
                with open('config_training.json', 'r') as config_training:
                    data = json.loads(config_training.read())
                    cfg_clp = int(data.get("clp", non_clp_start_round))
                    if cfg_clp > 0:
                        non_clp_start_round = cfg_clp
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                pass
            if self.non_clp_start_round is not None:
                non_clp_start_round = self.non_clp_start_round

            if server_round < non_clp_start_round:
                target_clients = min(3, available_count)
                guaranteed = 1
                # Sort known clients by ID descending to get highest IDs first
                known = []
                for proxy in available:
                    client_id = self.client_ids_by_cid.get(proxy.cid)
                    if client_id is not None:
                        known.append((client_id, proxy))
                known.sort(key=lambda item: item[0], reverse=True)  # descending → highest IDs first
                # pinned = [proxy for _, proxy in known[:guaranteed]]  # original: takes absolute highest IDs (e.g. 9, 8)
                # pinned = [proxy for _, proxy in known[1:guaranteed+1]]  # skip top 1, take next `guaranteed` (e.g. 8, 7)
                pinned = [proxy for _, proxy in known[2:guaranteed+2]]  # skip top 2, take next `guaranteed` (e.g. 7, 6)
                remaining = [p for p in available if p not in pinned]
                extra_needed = target_clients - len(pinned)
                if extra_needed > 0 and len(remaining) > 0:
                    pick_count = min(extra_needed, len(remaining))
                    pick_idx = np.random.choice(len(remaining), size=pick_count, replace=False)
                    pinned.extend([remaining[i] for i in pick_idx])
                clients = pinned
                selected_ids = [self.client_ids_by_cid.get(p.cid, -1) for p in clients]
                print(
                    f"[configure_fit] round={server_round}, clp={non_clp_start_round}, policy=clp_aware/15_top5_high_ids, "
                    f"num_clients={len(clients)}, selected_client_ids={selected_ids}"
                )
            else:
                target_clients = min(3, available_count)
                pick_idx = np.random.choice(available_count, size=target_clients, replace=False)
                clients = [available[i] for i in pick_idx]
                selected_ids = [self.client_ids_by_cid.get(p.cid, -1) for p in clients]
                print(
                    f"[configure_fit] round={server_round}, clp={non_clp_start_round}, policy=clp_aware/random_8, "
                    f"num_clients={len(clients)}, selected_client_ids={selected_ids}"
                )
        else:
            # cfl policy: conventional FL sampling without CLP-dependent switching.
            target_clients = min(self.cfl_clients_per_round, available_count)
            pick_idx = np.random.choice(available_count, size=target_clients, replace=False)
            clients = [available[i] for i in pick_idx]
            selected_ids = [self.client_ids_by_cid.get(p.cid, -1) for p in clients]
            print(
                f"[configure_fit] round={server_round}, policy=cfl/random, "
                f"cfl_clients_per_round={self.cfl_clients_per_round}, "
                f"num_clients={len(clients)}, selected_client_ids={selected_ids}"
            )

        # Record round start time just before dispatching to clients.
        # aggregate_fit will read this to compute the full round latency
        # (client training + gRPC transfer + server aggregation).
        self._round_start[server_round] = time.perf_counter()
        self._round_selected_count[server_round] = len(clients)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients
        
    """ 
    aggregate_fit():
        is responsible for aggregating the results returned by the clients 
        that were selected and asked to train in configure_fit().
        
    Reference: 
        https://flower.ai/docs/framework/how-to-implement-strategies.html#the-aggregate-fit-method
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,#
        # configs, # commented on 22/05 when we add configure_fit()
    ) -> fl.common.Parameters: # Parameters(tensors, tensor_type) - Model parameters.
        
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}     
           
        _received_bytes = sum(
            sum(len(t) for t in fit_res.parameters.tensors)
            for _, fit_res in results
        )

        _agg_t0 = time.perf_counter()
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        _agg_latency = time.perf_counter() - _agg_t0
        # Full round latency = time from configure_fit dispatch to aggregation complete
        # (covers client local training + gRPC transfer + server FedAvg computation)
        _round_latency = time.perf_counter() - self._round_start.get(server_round, _agg_t0)

        if aggregated_weights is not None:
            # get num_rounds from config_training json file to be use to verify
            # if the current round is the first round
            with open('config_training.json', 'r') as config_training:
                config=config_training.read()
                data = json.loads(config)
                num_rounds   = data['num_rounds']
                session      = data['session']
                clp          = data['clp']
                session_dir  = data.get('session_dir', f'Session-{session}')  # fallback for legacy runs

            session_path = f"../Server/model_weights/{session_dir}"
            if not os.path.exists(session_path):
                os.makedirs(session_path)

            if server_round < num_rounds:
                np.save(f"{session_path}/round-{server_round}-weights.npy", aggregated_weights)
            elif server_round == num_rounds:
                np.save(f"{session_path}/Global_model_seed{SEED}.npy", aggregated_weights)
                file_path = f"{session_path}/Global_model_seed{SEED}.npy"
                # Time the SHA-256 hash computation — pure blockchain-prep overhead
                _hash_t0 = time.perf_counter()
                with open(file_path, "rb") as f:
                    _file_bytes = f.read()
                    readable_hash = hashlib.sha256(_file_bytes).hexdigest()
                _hash_latency = time.perf_counter() - _hash_t0
                print(readable_hash)
                # Log hash computation as a blockchain overhead event;
                # SHA-256 hexdigest = 64 hex chars = 64 ASCII bytes (on-chain string size)
                self.latency_comm_logger.log(
                    part="blockchain",
                    event="hash_computation",
                    round_num=server_round,
                    latency_s=_hash_latency,
                    comm_bytes=len(readable_hash.encode()),
                )
                global_model_BC = blockchainService.addModel(session,server_round,file_path,readable_hash)

                # put the global model to the IPFS via pinata
                pinata_name = (
                    f"Global_model_{data['dataset']}_{data.get('model_type', 'cnn')}"
                    f"_seed{SEED}_{data['num_clients']}clients"
                    f"_{data['client_selection']}_alpha{data['alpha']}"
                    f"_{data['run_stamp']}.npy"
                )
                pinata.pin_file_to_ipfs(
                    path_to_file=file_path,
                    ipfs_destination_path='',
                    save_absolute_paths=False,
                    options={"pinataMetadata": {"name": pinata_name}},
                )
        # loop through the results and update contribution (pairs of key, value) where
        # the key is the client id and the value is a dict of data size, sent size
        # and num_rounds_participated: updated value
        # total_data_size = 0
        round_client_ids: List[int] = []
        for res in results:
            # results: List[Tuple[ClientProxy, FitRes]]
            # FitRes: parameters: Parameters , num_examples: int , metrics: Optional[Metrics] = None
            proxy, fit_res = res
            client_id = int(fit_res.metrics["client_id"])
            self.client_ids_by_cid[proxy.cid] = client_id
            round_client_ids.append(client_id)
            print(
                f"client id={client_id}, "
                f"client address={fit_res.metrics['client_address']}, "
                f"data size={fit_res.num_examples}"
            )

            if client_id not in self.contribution.keys():
                self.contribution[client_id]={
                    "data_size":fit_res.num_examples,
                    "num_rounds_participated":1,
                    "client_address":fit_res.metrics['client_address'],
                    # Update on 24/10/2024 - Track the round that the client joined to calculate the reward
                    "joined_round": [server_round]
                }
                self.contribution['total_data_size'] = self.contribution['total_data_size'] + fit_res.num_examples
            else:
                self.contribution[client_id]["num_rounds_participated"] += 1
                # Update on 24/10/2024 - Track the round that the client joined to calculate the reward
                self.contribution[client_id]["joined_round"].append(server_round)
        joined_ids_unique = sorted(set(round_client_ids))
        self.result['joined_client_ids'][server_round] = joined_ids_unique
        print(f"[round {server_round}] joined_client_ids={joined_ids_unique}")

        # ── CLP Detection ──────────────────────────────────────────────────────
        # Compute weighted average of client CLP-loss values (FGN):
        #   fgn(t) = Σ_i [ (n_i / N_total) * (lr * ||∇L_i(batch)||²) ]
        # Then detect a CLP via: (fgn(t) - fgn(t-1)) / fgn(t-1) >= threshold
        total_examples = sum(res[1].num_examples for res in results)
        if total_examples > 0:
            fgn_t = sum(
                (res[1].num_examples / total_examples) * res[1].metrics.get('clp_loss', 0.0)
                for res in results
            )
        else:
            fgn_t = 0.0

        self.fgn[server_round] = fgn_t

        fgn_prev = self.fgn.get(server_round - 1, 0.0)
        if fgn_prev > 0.0:
            relative_change = (fgn_t - fgn_prev) / fgn_prev
            self.is_clp[server_round] = relative_change >= self.threshold
        else:
            relative_change = None          # undefined at round 1 (no previous FGN)
            self.is_clp[server_round] = False

        self.result['fgn_relative_change'][server_round] = relative_change

        # Dynamic CLP -> non-CLP transition:
        # When relative change falls below threshold, the next round is the
        # first non-CLP round used by configure_fit for client selection.
        if (
            self.non_clp_start_round is None
            and relative_change is not None
            and relative_change < self.threshold
        ):
            self.non_clp_start_round = server_round + 1
            print(
                f"[CLP] auto non_clp_start_round set to {self.non_clp_start_round} "
                f"(triggered at round {server_round}, rel_change={relative_change:+.6f}, "
                f"threshold={self.threshold})"
            )

        rc_str = f"{relative_change:+.6f}" if relative_change is not None else "N/A (first round)"
        clp_marker = "  *** CLP DETECTED ***" if self.is_clp[server_round] else ""
        print(
            f"[CLP] round={server_round}"
            f"  fgn={fgn_t:.6f}"
            f"  prev_fgn={fgn_prev:.6f}"
            f"  rel_change={rc_str}"
            f"  threshold={self.threshold}"
            f"  is_CLP={self.is_clp[server_round]}"
            f"{clp_marker}"
        )

        # Append one JSON line per round to a dedicated FGN log file
        log_dir = data.get('log_dir', '../report/temp')
        os.makedirs(log_dir, exist_ok=True)
        fgn_log_path = os.path.join(log_dir, f'fgn_log_session{session}.jsonl')
        fgn_entry = {
            'round':        server_round,
            'fgn':          fgn_t,
            'prev_fgn':     fgn_prev,
            'rel_change':   relative_change,
            'threshold':    self.threshold,
            'is_clp':       self.is_clp[server_round],
            'joined_client_ids': joined_ids_unique,
        }
        with open(fgn_log_path, 'a') as f:
            f.write(json.dumps(fgn_entry) + '\n')
        # ───────────────────────────────────────────────────────────────────────

        # ── Latency & communication logging ────────────────────────────────────
        # Training: aggregation latency + bytes exchanged with clients.
        # Server sends the global model once per selected client.
        if aggregated_weights is not None and aggregated_weights[0] is not None:
            _model_bytes = sum(len(t) for t in aggregated_weights[0].tensors)
            _selected_clients = self._round_selected_count.get(server_round, len(results))
            _sent_bytes = _model_bytes * _selected_clients
        else:
            _model_bytes = 0
            _sent_bytes = 0

        self.latency_comm_logger.log(
            part="training",
            event="round",
            round_num=server_round,
            latency_s=_round_latency,
            comm_bytes=_received_bytes + _sent_bytes,
            received_bytes=_received_bytes,
            sent_bytes=_sent_bytes,
            model_bytes_per_client=_model_bytes,
            aggregation_latency_s=round(_agg_latency, 6),
            num_clients=len(results),
        )

        # Blockchain: addModel is called only on the final round; for all other
        # rounds log addWeight latency from the most recent blockchain tx.
        #   comm_bytes  = actual ABI-encoded calldata bytes sent to the chain node
        #   model_bytes = raw .npy file size (FL model size that drove this TX)
        bc_txs = blockchainService.metrics.transactions
        if bc_txs:
            last_bc = bc_txs[-1]
            if last_bc.get("function") in ("addModel", "addWeight"):
                npy_path = (
                    f"../Server/model_weights/{data.get('session_dir', '')}"
                    f"/Global_model_seed{SEED}.npy"
                    if server_round == data.get('num_rounds')
                    else f"../Server/model_weights/{data.get('session_dir', '')}"
                         f"/round-{server_round}-weights.npy"
                )
                model_bytes = os.path.getsize(npy_path) if os.path.exists(npy_path) else 0
                self.latency_comm_logger.log(
                    part="blockchain",
                    event=last_bc["function"],
                    round_num=server_round,
                    latency_s=last_bc.get("latency_s", 0.0),
                    comm_bytes=last_bc.get("calldata_bytes", 0),
                    model_bytes=model_bytes,
                )
        # Release per-round temporary bookkeeping once consumed.
        self._round_start.pop(server_round, None)
        self._round_selected_count.pop(server_round, None)
        # ───────────────────────────────────────────────────────────────────────

        return aggregated_weights
    
    """
    aggregate_evaluate():
        is responsible for aggregating the results returned by the clients
        that were selected and asked to evaluate in configure_evaluate().
    Reference: https://flower.ai/docs/framework/how-to-implement-strategies.html#the-aggregate-evaluate-method
    """
    def aggregate_evaluate(
        self,
        server_round: int,
        results, # : List[Tuple[ClientProxy, EvaluateRes]]
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        '''Ref: https://flower.ai/docs/framework/how-to-aggregate-evaluation-results.html#aggregate-custom-evaluation-results'''
        # Aggregate evaluation accuracy using weighted average
        if not results:
            return None, {}
        
        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")
        print(f"Round {server_round} loss aggregated from client results: {aggregated_loss}")
        
        self.result['aggregated_loss'][server_round]=aggregated_loss
        self.result['aggregated_accuracy'][server_round]=aggregated_accuracy

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}

'''Reference: 
[1]. https://flower.ai/docs/framework/explanation-federated-evaluation.html#built-in-strategies
'''
def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    dataset = normalize_dataset_name(DATASET_NAME)

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    input_shape = get_dataset_input_shape(dataset)
    (x_train, y_train), _ = load_dataset_raw(dataset)

    h, w, c = input_shape

    random_indices = np.random.choice(x_train.shape[0], int(x_train.shape[0] * 0.2), replace=False)
    x_val, y_val = x_train[random_indices], y_train[random_indices].reshape(-1)
    x_val = x_val.reshape(-1, h, w, c)
    x_val = x_val/255.
    
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
        return loss, {"accuracy": accuracy}

    return evaluate


''' The server can pass new configuration values to the client each round using this function
- The client will receive the dictionary returned by the on_fit_config_fn in its own client.fit() function.
'''
def get_on_fit_config_fn() -> Callable[[int], Dict[str, Any]]:
    """Return training configuration dict for each round.
    Uses batch size/local epochs from config_training.json.
    """

    def fit_config(server_round: int) -> Dict[str, str]:
        with open('config_training.json', 'r') as config_training:
            data = json.loads(config_training.read())
            session     = data['session']
            clp         = data['clp']
            session_dir = data.get('session_dir', f'Session-{session}')
            batch_size  = int(data.get('batch_size', 32))
            local_epochs = int(data.get('local_epochs', 2))

        config = {
            "batch_size":    max(1, batch_size),
            "local_epochs":  max(1, local_epochs),
            "round":         server_round,
            "session":       session,
            "session_dir":   session_dir,
            "clp":           clp,
        }
        return config
        
    return fit_config


def evaluate_config(server_round: int):
    """ Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    # val_steps = 5 if server_round < 32 else 10
    val_steps = 3
    with open('config_training.json', 'r') as config_training:
        data = json.loads(config_training.read())
        session     = data['session']
        session_dir = data.get('session_dir', f'Session-{session}')
    return {"val_steps": val_steps, "round": server_round, "session": session, "session_dir": session_dir}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
