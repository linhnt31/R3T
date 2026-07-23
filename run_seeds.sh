#!/usr/bin/env bash
# Run FL experiments across multiple seeds sequentially.
# Each seed gets a fresh blockchain deployment and a full server+client run.
#
# Usage:
#   ./run_seeds.sh
#   DATASET=cifar10 NUM_CLIENTS=10 ./run_seeds.sh   # override defaults via env vars

set -euo pipefail

# ── Conda environment ──────────────────────────────────────────────────────────
CONDA_ENV="blockfed"
# Source conda so that 'conda activate' works inside the script
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Configurable parameters ────────────────────────────────────────────────────
SEEDS=(42)
# SEEDS=(42 1234 20 7 41 100 31)
# SEEDS=(8)
ALPHAS=(0.1 0.4 0.7)
DATASETS=(fmnist cifar10)
# DATASETS=(fmnist)
NUM_CLIENTS="${NUM_CLIENTS:-10}"
NUM_ROUNDS="${NUM_ROUNDS:-100}"
BUDGET="${BUDGET:-5000}"
CLP="${CLP:-10}"
THRESHOLD="${THRESHOLD:-0.05}"
CLIENT_SELECTIONS=(clp_aware cfl)
# CLIENT_SELECTIONS=(clp_aware)
MODEL_TYPE="${MODEL_TYPE:-cnn}"                        # cnn or alexnet
RATIO="${RATIO:-1.4}"                                      # geometric growth ratio for per-client sample counts
BATCH_SIZE="${BATCH_SIZE:-32}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
CLIENT_LAUNCH_DELAY="${CLIENT_LAUNCH_DELAY:-0.2}"

# Seconds to wait after starting the server before launching clients
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-8}"
# Seconds to wait for Ganache to be ready after restart
GANACHE_STARTUP_WAIT="${GANACHE_STARTUP_WAIT:-5}"
GANACHE_PORT="${GANACHE_PORT:-8545}"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
BLOCKCHAIN_DIR="$REPO_ROOT/Blockchain"
SERVER_DIR="$REPO_ROOT/Server"
CLIENT_DIR="$REPO_ROOT/Client"
# ──────────────────────────────────────────────────────────────────────────────

log() { echo "[run_seeds] $*"; }

GANACHE_PID=""
TOTAL_RUNS=$((${#CLIENT_SELECTIONS[@]} * ${#DATASETS[@]} * ${#ALPHAS[@]} * ${#SEEDS[@]}))
RUN_NUMBER=0

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping server (pid=$SERVER_PID) …"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

cleanup_ganache() {
    if [[ -n "${GANACHE_PID:-}" ]] && kill -0 "$GANACHE_PID" 2>/dev/null; then
        log "Stopping Ganache (pid=$GANACHE_PID) …"
        kill "$GANACHE_PID" 2>/dev/null || true
        wait "$GANACHE_PID" 2>/dev/null || true
    fi
    GANACHE_PID=""
}

restart_ganache() {
    cleanup_ganache
    log "Starting fresh Ganache on port $GANACHE_PORT …"
    ganache --deterministic --accounts $((NUM_CLIENTS + 1)) --defaultBalanceEther $((BUDGET)) --port "$GANACHE_PORT" &
    GANACHE_PID=$!
    sleep "$GANACHE_STARTUP_WAIT"
    if ! kill -0 "$GANACHE_PID" 2>/dev/null; then
        log "ERROR: Ganache failed to start."
        exit 1
    fi
    log "Ganache ready (pid=$GANACHE_PID)."
}

trap 'log "Interrupted — cleaning up …"; cleanup_server; cleanup_ganache; exit 1' INT TERM

# ── Already-completed experiments (skip these) ─────────────────────────────────
# Format: "CLIENT_SELECTION:DATASET:ALPHA:SEED"
DONE_EXPERIMENTS=()

is_done() {
    local key="$1"
    for entry in "${DONE_EXPERIMENTS[@]}"; do
        [[ "$entry" == "$key" ]] && return 0
    done
    return 1
}
# ──────────────────────────────────────────────────────────────────────────────

for CLIENT_SELECTION in "${CLIENT_SELECTIONS[@]}"; do
for DATASET in "${DATASETS[@]}"; do
for ALPHA in "${ALPHAS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    RUN_NUMBER=$((RUN_NUMBER + 1))

    if is_done "${CLIENT_SELECTION}:${DATASET}:${ALPHA}:${SEED}"; then
        log "Skipping already-completed: selection=$CLIENT_SELECTION  dataset=$DATASET  alpha=$ALPHA  seed=$SEED"
        continue
    fi

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Starting experiment  run=$RUN_NUMBER/$TOTAL_RUNS  dataset=$DATASET  alpha=$ALPHA  seed=$SEED  clients=$NUM_CLIENTS  selection=$CLIENT_SELECTION"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 1. Restart Ganache (resets all ETH balances to 1000) then redeploy contracts
    restart_ganache
    log "Deploying contracts …"
    (cd "$BLOCKCHAIN_DIR" && truffle migrate --reset --network development)
    log "Contracts deployed."

    # 2. Start the FL server in the background
    SERVER_LOG="$REPO_ROOT/report/temp/server_${DATASET}_${MODEL_TYPE}_seed${SEED}_${NUM_CLIENTS}clients_${CLIENT_SELECTION}_alpha${ALPHA}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$SERVER_LOG")"
    log "Starting server … (log: $SERVER_LOG)"
    (cd "$SERVER_DIR" && python server.py \
        --num_rounds "$NUM_ROUNDS" \
        --budget "$BUDGET" \
        --clp "$CLP" \
        --threshold "$THRESHOLD" \
        --num_clients "$NUM_CLIENTS" \
        --seed "$SEED" \
        --dataset "$DATASET" \
        --alpha "$ALPHA" \
        --batch_size "$BATCH_SIZE" \
        --local_epochs "$LOCAL_EPOCHS" \
        --model_type "$MODEL_TYPE" \
        --ratio "$RATIO" \
        --"$CLIENT_SELECTION" \
    ) > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    log "Server started (pid=$SERVER_PID). Waiting ${SERVER_STARTUP_WAIT}s …"
    sleep "$SERVER_STARTUP_WAIT"

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ERROR: Server exited prematurely. Check $SERVER_LOG"
        exit 1
    fi

    # 3. Launch all clients and wait for them to finish
    log "Launching $NUM_CLIENTS clients …"
    (cd "$CLIENT_DIR" && python launch_clients.py \
        --num_clients "$NUM_CLIENTS" \
        --server_address "$SERVER_ADDRESS" \
        --seed "$SEED" \
        --dataset "$DATASET" \
        --alpha "$ALPHA" \
        --delay "$CLIENT_LAUNCH_DELAY" \
    )
    log "All clients finished for run=$RUN_NUMBER/$TOTAL_RUNS  seed=$SEED."

    # 4. Wait for the server to finish cleanly
    log "Waiting for server to finish …"
    wait "$SERVER_PID" || true
    SERVER_PID=""
    log "Server finished for run=$RUN_NUMBER/$TOTAL_RUNS  seed=$SEED."
done
done
done
done

cleanup_ganache
log "All experiments complete: total_runs=$TOTAL_RUNS  datasets=${DATASETS[*]}  alphas=${ALPHAS[*]}  seeds=${SEEDS[*]}  selections=${CLIENT_SELECTIONS[*]}"
