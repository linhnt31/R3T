"""
Blockchain overhead metrics collector.

Records every on-chain transaction (latency, gas, block), then computes
aggregate statistics (mean/std/min/max latency, total gas, TPS).

Usage
-----
# In blockchain_service.py:
    metrics = BlockchainMetricsCollector(role="server")
    with metrics.record("addStrategy"):
        tx_hash = contract.functions.addStrategy(...).transact(...)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    metrics.attach_receipt(receipt)

# At the end of training:
    path = metrics.save("../report/temp")
"""

import json
import math
import os
import time
from contextlib import contextmanager
from typing import Optional


class BlockchainMetricsCollector:
    """
    Captures per-transaction blockchain overhead and derives aggregate stats.

    Attributes
    ----------
    role : str
        Label for this participant, e.g. "server" or "client_3".
    transactions : list[dict]
        One entry per confirmed transaction.
    """

    def __init__(self, role: str):
        self.role = role
        self.transactions: list = []
        self._pending: Optional[dict] = None   # open context-manager slot
        self.balance_snapshots: list = []       # [{label, wall_time, balances}]

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    @contextmanager
    def record(self, function_name: str):
        """
        Context manager that times the block and stores function name.
        Call attach_receipt() right after to save gas / block data.

        Example
        -------
            with metrics.record("addWeight"):
                tx_hash = contract.functions.addWeight(...).transact(...)
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            metrics.attach_receipt(receipt)
        """
        entry = {
            "function":     function_name,
            "wall_time":    time.time(),        # absolute UNIX timestamp
            "latency_s":    None,
            "gas_used":     None,
            "block_number": None,
            "tx_hash":      None,
            "status":       None,
        }
        t0 = time.perf_counter()
        try:
            yield entry
        finally:
            entry["latency_s"] = time.perf_counter() - t0
            self._pending = entry
            self.transactions.append(entry)

    def attach_receipt(self, receipt) -> None:
        """
        Fill in gas / block data from a web3 transaction receipt.
        Must be called immediately after the ``with metrics.record(...)`` block.
        """
        if self._pending is None:
            return
        self._pending["gas_used"]     = receipt.get("gasUsed",     receipt.get("gas_used"))
        self._pending["block_number"] = receipt.get("blockNumber",  receipt.get("block_number"))
        self._pending["status"]       = receipt.get("status", 1)
        raw_hash = receipt.get("transactionHash") or receipt.get("transaction_hash")
        if raw_hash is not None:
            self._pending["tx_hash"] = raw_hash.hex() if hasattr(raw_hash, "hex") else str(raw_hash)
        self._pending = None

    def attach_extras(self, data: dict) -> None:
        """
        Merge arbitrary extra fields into the most recently completed transaction.
        Useful for recording payment amounts, client balances, or any domain data
        that is not part of the standard receipt.

        Call this after ``attach_receipt``.

        Example
        -------
            self.metrics.attach_receipt(receipt)
            self.metrics.attach_extras({
                "payment_eth":        float(payment),
                "client_address":     _client_address,
                "balance_before_wei": balance_before,
                "balance_after_wei":  balance_after,
            })
        """
        if not self.transactions:
            return
        self.transactions[-1].update(data)

    def record_balances(self, label: str, balances: dict) -> None:
        """
        Store a named balance snapshot for all clients.

        Parameters
        ----------
        label : str
            Human-readable label, e.g. ``"before_payment"`` or ``"after_payment"``.
        balances : dict
            Mapping of identifier → ``{"address": str, "balance_wei": int}``.
            The ``balance_eth`` field is derived automatically.
        """
        snapshot = {"label": label, "wall_time": time.time(), "balances": {}}
        for key, info in balances.items():
            wei = info["balance_wei"]
            snapshot["balances"][str(key)] = {
                "address":     info["address"],
                "balance_wei": wei,
                "balance_eth": round(wei / 1e18, 8),
            }
        self.balance_snapshots.append(snapshot)

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    def _stats_of(self, values: list) -> dict:
        """Return mean / std / min / max for a list of numbers."""
        n = len(values)
        if n == 0:
            return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        return {
            "count": n,
            "mean":  round(mean,     6),
            "std":   round(math.sqrt(variance), 6),
            "min":   round(min(values), 6),
            "max":   round(max(values), 6),
        }

    def compute_stats(self) -> dict:
        """
        Return a dict with:
          - overall latency, gas, TPS
          - per-function breakdown of the same metrics
          - full transaction log
        """
        txs = self.transactions
        if not txs:
            return {"role": self.role, "total_transactions": 0}

        latencies  = [t["latency_s"] for t in txs if t["latency_s"] is not None]
        gas_values = [t["gas_used"]  for t in txs if t["gas_used"]  is not None]

        # Throughput: tx / (last_wall_time - first_wall_time)
        wall_times = [t["wall_time"] for t in txs]
        span = wall_times[-1] - wall_times[0] if len(wall_times) > 1 else latencies[0]
        tps  = len(txs) / span if span > 0 else 0.0
        # Processing TPS: tx / sum(latencies) — excludes idle time between rounds
        total_latency = sum(latencies)
        tps_processing = len(txs) / total_latency if total_latency > 0 else 0.0

        # Per-function breakdown
        fn_buckets: dict = {}
        for tx in txs:
            fn = tx["function"]
            fn_buckets.setdefault(fn, {"latencies": [], "gas": [], "count": 0})
            fn_buckets[fn]["count"] += 1
            if tx["latency_s"] is not None:
                fn_buckets[fn]["latencies"].append(tx["latency_s"])
            if tx["gas_used"] is not None:
                fn_buckets[fn]["gas"].append(tx["gas_used"])

        per_function = {}
        for fn, data in fn_buckets.items():
            lat_stats = self._stats_of(data["latencies"])
            gas_stats = self._stats_of(data["gas"])
            per_function[fn] = {
                "call_count":       data["count"],
                "latency_s":        lat_stats,
                "gas":              gas_stats,
                "total_gas":        sum(data["gas"]),
                "throughput_tps":   round(
                    data["count"] / sum(data["latencies"]) if data["latencies"] else 0, 4
                ),
            }

        # Payment / balance summary (only present when attach_extras was used)
        payment_summary = self._compute_payment_summary(txs)

        result = {
            "role":               self.role,
            "total_transactions": len(txs),
            "observation_span_s":      round(span, 4),
            "throughput_tps":          round(tps, 4),
            "throughput_tps_processing": round(tps_processing, 4),
            "latency_s":               self._stats_of(latencies),
            "gas":                {
                **self._stats_of(gas_values),
                "total": sum(gas_values),
            },
            "per_function":       per_function,
            "transactions":       txs,   # full raw log
        }
        if payment_summary:
            result["payments"] = payment_summary
        if self.balance_snapshots:
            result["client_balances"] = self.balance_snapshots
        return result

    def _compute_payment_summary(self, txs: list) -> dict:
        """
        Aggregate payment_eth and balance snapshots across transactions.
        Returns an empty dict if no payment data was recorded.
        """
        payment_txs = [t for t in txs if "payment_eth" in t]
        if not payment_txs:
            return {}

        total_eth = sum(t["payment_eth"] for t in payment_txs)

        # Per-client breakdown (keyed by client_address if present)
        per_client: dict = {}
        for t in payment_txs:
            addr = t.get("client_address", "unknown")
            if addr not in per_client:
                per_client[addr] = {
                    "rounds_paid":         0,
                    "total_paid_eth":      0.0,
                    "initial_balance_wei": t.get("balance_before_wei"),
                    "final_balance_wei":   None,
                }
            per_client[addr]["rounds_paid"]    += 1
            per_client[addr]["total_paid_eth"] += t["payment_eth"]
            if t.get("balance_after_wei") is not None:
                per_client[addr]["final_balance_wei"] = t["balance_after_wei"]

        for addr, info in per_client.items():
            info["total_paid_eth"] = round(info["total_paid_eth"], 8)

        return {
            "total_eth":  round(total_eth, 8),
            "num_payments": len(payment_txs),
            "per_client": per_client,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, log_dir: str = "../report/temp", run_tag: Optional[str] = None) -> str:
        """
        Write stats to:
          <log_dir>/blockchain_metrics_<role>.json
        or, if run_tag is provided:
          <log_dir>/blockchain_metrics_<role>_<run_tag>.json
        and print a console summary.  Returns the output path.
        """
        os.makedirs(log_dir, exist_ok=True)
        stats = self.compute_stats()
        suffix = f"_{run_tag}" if run_tag else ""
        path  = os.path.join(log_dir, f"blockchain_metrics_{self.role}{suffix}.json")

        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

        self._print_summary(stats)
        print(f"[metrics] Full report saved → {path}")
        return path

    def _print_summary(self, stats: dict) -> None:
        n   = stats.get("total_transactions", 0)
        tps            = stats.get("throughput_tps", 0)
        tps_processing = stats.get("throughput_tps_processing", 0)
        lat = stats.get("latency_s", {})
        gas = stats.get("gas", {})
        print(
            f"\n{'─'*55}\n"
            f" Blockchain overhead — {stats['role']}\n"
            f"{'─'*55}\n"
            f"  Transactions : {n}\n"
            f"  Throughput   : {tps:.4f} TPS  (processing: {tps_processing:.4f} TPS)\n"
            f"  Latency (s)  : mean={lat.get('mean',0):.4f}  "
            f"std={lat.get('std',0):.4f}  "
            f"min={lat.get('min',0):.4f}  "
            f"max={lat.get('max',0):.4f}\n"
            f"  Gas          : total={gas.get('total',0)}  "
            f"mean={gas.get('mean',0):.0f}  "
            f"min={gas.get('min',0)}  "
            f"max={gas.get('max',0)}\n"
            f"{'─'*55}"
        )
        for fn, d in stats.get("per_function", {}).items():
            l = d["latency_s"]
            print(
                f"  {fn:<28} calls={d['call_count']:>3}  "
                f"lat_mean={l['mean']:.4f}s  gas_total={d['total_gas']}"
            )

        snapshots = stats.get("client_balances", [])
        if snapshots:
            # Show a diff table if both before/after snapshots exist
            before = next((s for s in snapshots if "before" in s["label"]), None)
            after  = next((s for s in snapshots if "after"  in s["label"]), None)
            if before and after:
                print(f"\n  Client balances (before → after payment)")
                for key in before["balances"]:
                    b = before["balances"][key]
                    a = after["balances"].get(key, {})
                    b_eth = b["balance_eth"]
                    a_eth = a.get("balance_eth", b_eth)
                    delta = a_eth - b_eth
                    print(
                        f"    {key:<12}  {b_eth:.4f} → {a_eth:.4f} ETH  "
                        f"(Δ{delta:+.4f})"
                    )
            else:
                for snap in snapshots:
                    print(f"\n  Balance snapshot [{snap['label']}]")
                    for key, info in snap["balances"].items():
                        print(f"    {key:<12}  {info['balance_eth']:.4f} ETH")

        pay = stats.get("payments")
        if pay:
            print(
                f"\n  Payments ({pay['num_payments']} txs)  total={pay['total_eth']:.6f} ETH"
            )
            for addr, info in pay.get("per_client", {}).items():
                short_addr = addr[:10] + "…" if len(addr) > 10 else addr
                bal_before = info.get("initial_balance_wei")
                bal_after  = info.get("final_balance_wei")
                bal_str = ""
                if bal_before is not None and bal_after is not None:
                    delta = (bal_after - bal_before) / 1e18
                    bal_str = (
                        f"  balance: {bal_before/1e18:.4f}→{bal_after/1e18:.4f} ETH "
                        f"(Δ{delta:+.4f})"
                    )
                print(
                    f"    {short_addr}  rounds={info['rounds_paid']}  "
                    f"paid={info['total_paid_eth']:.6f} ETH{bal_str}"
                )
        print()


# ---------------------------------------------------------------------------
# Shared stats helper (used by both BlockchainMetricsCollector and
# LatencyCommLogger so the logic is not duplicated)
# ---------------------------------------------------------------------------

def _stats_helper(values: list) -> dict:
    """Return count / mean / std / min / max for a list of numbers."""
    n = len(values)
    if n == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / n
    std  = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {
        "count": n,
        "mean":  round(mean, 6),
        "std":   round(std,  6),
        "min":   round(min(values), 6),
        "max":   round(max(values), 6),
    }


# ---------------------------------------------------------------------------
# Latency & communication logger (training vs blockchain, per round)
# ---------------------------------------------------------------------------

class LatencyCommLogger:
    """
    Logs per-round latency and communication bytes for the *training* and
    *blockchain* parts separately.

    Each ``log()`` call buffers one entry.  Call ``save()`` at the end of
    training to flush everything to two files in *log_dir*:

    * ``latency_comm_<role>[_<run_tag>].jsonl``        — raw per-event JSONL
    * ``latency_comm_<role>[_<run_tag>]_summary.json`` — aggregate stats

    Usage
    -----
    # create once per role (server / client_N):
        lc = LatencyCommLogger(role="client_3")

    # inside fit() / aggregate_fit():
        lc.log("training",    "fit",           round_num=r,
                latency_s=1.2, comm_bytes=4_000_000)
        lc.log("blockchain",  "addWeight",     round_num=r,
                latency_s=0.3, comm_bytes=120_000)

    # at the end of training:
        lc.save(log_dir="../report/temp/run_xyz", run_tag="20260406_120000")
    """

    def __init__(self, role: str):
        self.role = role
        self.entries: list = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def log(self, part: str, event: str, round_num: int,
            latency_s: float, comm_bytes: int, **extra) -> None:
        """
        Buffer one latency + communication measurement.

        Parameters
        ----------
        part      : ``"training"`` or ``"blockchain"``
        event     : human-readable label, e.g. ``"fit"``, ``"aggregate_fit"``,
                    ``"addWeight"``
        round_num : FL round number (use 0 for non-round events)
        latency_s : elapsed wall-clock seconds for this event
        comm_bytes: bytes transferred / written for this event
        **extra   : arbitrary extra fields (``client_id``,
                    ``download_bytes``, ``upload_bytes``, …)
        """
        entry = {
            "wall_time":  time.time(),
            "role":       self.role,
            "part":       part,
            "event":      event,
            "round":      round_num,
            "latency_s":  round(latency_s, 6),
            "comm_bytes": comm_bytes,
        }
        entry.update(extra)
        self.entries.append(entry)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, log_dir: str, run_tag: Optional[str] = None) -> str:
        """
        Flush buffered entries and write summary statistics.

        Returns the path to the summary JSON file.
        """
        os.makedirs(log_dir, exist_ok=True)
        suffix       = f"_{run_tag}" if run_tag else ""
        jsonl_path   = os.path.join(log_dir, f"latency_comm_{self.role}{suffix}.jsonl")
        summary_path = os.path.join(log_dir, f"latency_comm_{self.role}{suffix}_summary.json")

        # Append entries to the JSONL file
        with open(jsonl_path, "a") as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")

        # Compute summary from everything buffered this session
        summary = self._compute_summary(self.entries)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self._print_summary(summary)
        print(f"[latency_comm] Summary saved → {summary_path}")
        return summary_path

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _compute_summary(self, entries: list) -> dict:
        from collections import defaultdict
        # buckets[part][event] = {"latencies": [...], "comm": [...]}
        buckets: dict = defaultdict(lambda: defaultdict(
            lambda: {"latencies": [], "comm": []}
        ))
        for e in entries:
            b = buckets[e.get("part", "unknown")][e.get("event", "unknown")]
            if e.get("latency_s") is not None:
                b["latencies"].append(e["latency_s"])
            if e.get("comm_bytes") is not None:
                b["comm"].append(e["comm_bytes"])

        result: dict = {"role": self.role, "total_events": len(entries), "parts": {}}
        for part_name, events in buckets.items():
            part_result: dict = {}
            all_lat:  list = []
            all_comm: list = []
            for event_name, data in events.items():
                lats  = data["latencies"]
                comms = data["comm"]
                part_result[event_name] = {
                    "call_count":       len(lats),
                    "latency_s":        _stats_helper(lats),
                    "total_latency_s":  round(sum(lats), 6),
                    "comm_bytes":       _stats_helper(comms),
                    "total_comm_bytes": sum(comms),
                }
                all_lat.extend(lats)
                all_comm.extend(comms)
            part_result["_total"] = {
                "latency_s":        _stats_helper(all_lat),
                "total_latency_s":  round(sum(all_lat), 6),
                "comm_bytes":       _stats_helper(all_comm),
                "total_comm_bytes": sum(all_comm),
            }
            result["parts"][part_name] = part_result
        return result

    def _print_summary(self, summary: dict) -> None:
        print(f"\n{'─'*60}")
        print(f" Latency & Communication — {summary['role']}")
        print(f"{'─'*60}")
        for part_name, events in summary.get("parts", {}).items():
            total = events.get("_total", {})
            print(
                f"  [{part_name.upper()}]"
                f"  total_latency={total.get('total_latency_s', 0):.3f} s"
                f"  total_comm={total.get('total_comm_bytes', 0):,} bytes"
            )
            for event_name, data in events.items():
                if event_name == "_total":
                    continue
                l = data["latency_s"]
                print(
                    f"    {event_name:<30} calls={data['call_count']:>3}"
                    f"  lat_mean={l['mean']:.4f} s"
                    f"  comm_total={data['total_comm_bytes']:,} bytes"
                )
        print(f"{'─'*60}\n")


# ---------------------------------------------------------------------------
# Run-level aggregation (combines server + all client metric files)
# ---------------------------------------------------------------------------

def aggregate_run(run_dir: str, run_tag: Optional[str] = None) -> str:
    """
    Read every ``blockchain_metrics_*.json`` file in *run_dir*, merge all
    transactions, and compute run-level aggregate statistics:

    * overall latency (mean / std / min / max)
    * overall throughput (TPS) across the full observation window
    * overall gas (mean / std / min / max / total)
    * per-function breakdown  (addWeight, addContribution, addStrategy, …)
    * per-role summary        (server, client_0 … client_N)

    The result is written to::

        <run_dir>/blockchain_metrics_overall[_<run_tag>].json

    and a summary table is printed to stdout.  Returns the output path.
    """
    import glob as _glob

    pattern = os.path.join(run_dir, "blockchain_metrics_*.json")
    files = sorted(
        f for f in _glob.glob(pattern)
        if "_overall" not in os.path.basename(f)
    )
    if not files:
        print(f"[metrics] aggregate_run: no blockchain_metrics_*.json files found in {run_dir}")
        return ""

    # ── Collect all transactions keyed by role ──────────────────────────────
    all_txs: list = []
    per_role: dict = {}

    for path in files:
        with open(path) as f:
            data = json.load(f)
        role = data.get("role", os.path.basename(path))
        txs  = data.get("transactions", [])
        per_role[role] = data
        all_txs.extend(txs)

    if not all_txs:
        print("[metrics] aggregate_run: no transactions found across all files")
        return ""

    # ── Helper ──────────────────────────────────────────────────────────────
    def _stats(values: list) -> dict:
        n = len(values)
        if n == 0:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        mean = sum(values) / n
        std  = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
        return {
            "count": n,
            "mean":  round(mean, 6),
            "std":   round(std,  6),
            "min":   round(min(values), 6),
            "max":   round(max(values), 6),
        }

    # ── Overall latency & gas ───────────────────────────────────────────────
    latencies  = [t["latency_s"] for t in all_txs if t.get("latency_s") is not None]
    gas_values = [t["gas_used"]  for t in all_txs if t.get("gas_used")  is not None]

    # Observation window: earliest to latest wall_time across all roles
    wall_times = [t["wall_time"] for t in all_txs if t.get("wall_time") is not None]
    span = (max(wall_times) - min(wall_times)) if len(wall_times) > 1 else (latencies[0] if latencies else 0)
    tps  = len(all_txs) / span if span > 0 else 0.0
    # Processing TPS: tx / sum(latencies) — excludes idle time between rounds
    total_latency = sum(latencies)
    tps_processing = len(all_txs) / total_latency if total_latency > 0 else 0.0

    # ── Per-function breakdown ──────────────────────────────────────────────
    fn_buckets: dict = {}
    for tx in all_txs:
        fn = tx.get("function", "unknown")
        fn_buckets.setdefault(fn, {"latencies": [], "gas": [], "count": 0})
        fn_buckets[fn]["count"] += 1
        if tx.get("latency_s") is not None:
            fn_buckets[fn]["latencies"].append(tx["latency_s"])
        if tx.get("gas_used") is not None:
            fn_buckets[fn]["gas"].append(tx["gas_used"])

    per_function = {}
    for fn, data in fn_buckets.items():
        lat_stats = _stats(data["latencies"])
        gas_stats = _stats(data["gas"])
        per_function[fn] = {
            "call_count":     data["count"],
            "latency_s":      lat_stats,
            "gas":            gas_stats,
            "total_gas":      sum(data["gas"]),
            "throughput_tps": round(
                data["count"] / sum(data["latencies"]) if data["latencies"] else 0, 4
            ),
        }

    # ── Per-role summary (compact — no raw tx list) ─────────────────────────
    role_summary = {}
    for role, data in sorted(per_role.items()):
        role_summary[role] = {
            "total_transactions": data.get("total_transactions", 0),
            "observation_span_s": data.get("observation_span_s", 0),
            "throughput_tps":     data.get("throughput_tps", 0),
            "latency_s":          data.get("latency_s", {}),
            "gas":                data.get("gas", {}),
        }

    # ── Assemble result ─────────────────────────────────────────────────────
    result = {
        "roles_included":     sorted(per_role.keys()),
        "total_transactions":        len(all_txs),
        "observation_span_s":        round(span, 4),
        "throughput_tps":            round(tps, 4),
        "throughput_tps_processing": round(tps_processing, 4),
        "latency_s":                 _stats(latencies),
        "gas": {
            **_stats(gas_values),
            "total": sum(gas_values),
        },
        "per_function": per_function,
        "per_role":     role_summary,
    }

    # ── Save ────────────────────────────────────────────────────────────────
    suffix = f"_{run_tag}" if run_tag else ""
    out_path = os.path.join(run_dir, f"blockchain_metrics_overall{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Print summary ───────────────────────────────────────────────────────
    lat = result["latency_s"]
    gas = result["gas"]
    print(
        f"\n{'═'*60}\n"
        f" Blockchain overhead — OVERALL ({len(per_role)} roles)\n"
        f"{'═'*60}\n"
        f"  Transactions   : {result['total_transactions']}\n"
        f"  Obs. window    : {result['observation_span_s']:.1f} s\n"
        f"  Throughput     : {result['throughput_tps']:.4f} TPS  (processing: {result['throughput_tps_processing']:.4f} TPS)\n"
        f"  Latency (s)    : mean={lat['mean']:.4f}  std={lat['std']:.4f}"
        f"  min={lat['min']:.4f}  max={lat['max']:.4f}\n"
        f"  Gas            : total={gas['total']}  mean={gas['mean']:.0f}"
        f"  min={gas['min']}  max={gas['max']}\n"
        f"{'─'*60}"
    )
    print("  Per-function:")
    for fn, d in sorted(per_function.items()):
        l = d["latency_s"]
        print(
            f"    {fn:<28} calls={d['call_count']:>4}  "
            f"lat_mean={l['mean']:.4f}s  gas_total={d['total_gas']}"
        )
    print("  Per-role:")
    for role, d in sorted(role_summary.items()):
        l = d["latency_s"]
        print(
            f"    {role:<16} txs={d['total_transactions']:>4}  "
            f"lat_mean={l.get('mean', 0):.4f}s  "
            f"tps={d['throughput_tps']:.4f}"
        )
    print(f"{'═'*60}")
    print(f"[metrics] Overall report saved → {out_path}\n")
    return out_path


# ---------------------------------------------------------------------------
# Run-level latency & communication aggregation (combines server + all clients)
# ---------------------------------------------------------------------------

def aggregate_latency_comm_run(run_dir: str, run_tag: Optional[str] = None) -> str:
    """
    Read every ``latency_comm_*.jsonl`` file in *run_dir*, merge all entries,
    and compute a run-level latency & communication summary broken down by:

    * part  (``"training"`` vs ``"blockchain"``)
    * event (``"fit"``, ``"round"``, ``"addWeight"``, ``"hash_computation"``, …)
    * role  (``"server"``, ``"client_0"`` … ``"client_N"``)

    The result is written to::

        <run_dir>/latency_comm_overall[_<run_tag>].json

    and a summary table is printed to stdout.  Returns the output path.
    """
    import glob as _glob
    from collections import defaultdict

    pattern = os.path.join(run_dir, "latency_comm_*.jsonl")
    files = sorted(
        f for f in _glob.glob(pattern)
        if "_overall" not in os.path.basename(f)
    )
    if not files:
        print(f"[metrics] aggregate_latency_comm_run: no latency_comm_*.jsonl files in {run_dir}")
        return ""

    # ── Collect all entries keyed by role ──────────────────────────────────
    all_entries: list = []
    per_role_entries: dict = {}

    for path in files:
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if not entries:
            continue
        # Derive role from the "role" field inside the entries (most reliable)
        role = entries[0].get("role", os.path.basename(path))
        per_role_entries[role] = entries
        all_entries.extend(entries)

    if not all_entries:
        print("[metrics] aggregate_latency_comm_run: no entries found across all files")
        return ""

    # ── Shared aggregation helper ───────────────────────────────────────────
    def _compute_parts(entries: list) -> dict:
        buckets: dict = defaultdict(lambda: defaultdict(
            lambda: {"latencies": [], "comm": [], "model": []}
        ))
        for e in entries:
            b = buckets[e.get("part", "unknown")][e.get("event", "unknown")]
            if e.get("latency_s") is not None:
                b["latencies"].append(e["latency_s"])
            if e.get("comm_bytes") is not None:
                b["comm"].append(e["comm_bytes"])
            if e.get("model_bytes") is not None:
                b["model"].append(e["model_bytes"])

        parts: dict = {}
        for part_name, events in buckets.items():
            part_result: dict = {}
            all_lat, all_comm, all_model = [], [], []
            for event_name, data in events.items():
                lats  = data["latencies"]
                comms = data["comm"]
                models = data["model"]
                event_entry: dict = {
                    "call_count":       len(lats),
                    "latency_s":        _stats_helper(lats),
                    "total_latency_s":  round(sum(lats), 6),
                    "comm_bytes":       _stats_helper(comms),
                    "total_comm_bytes": sum(comms),
                }
                if models:
                    event_entry["model_bytes"]       = _stats_helper(models)
                    event_entry["total_model_bytes"]  = sum(models)
                part_result[event_name] = event_entry
                all_lat.extend(lats)
                all_comm.extend(comms)
                all_model.extend(models)

            total_entry: dict = {
                "latency_s":        _stats_helper(all_lat),
                "total_latency_s":  round(sum(all_lat), 6),
                "comm_bytes":       _stats_helper(all_comm),
                "total_comm_bytes": sum(all_comm),
            }
            if all_model:
                total_entry["model_bytes"]      = _stats_helper(all_model)
                total_entry["total_model_bytes"] = sum(all_model)
            part_result["_total"] = total_entry
            parts[part_name] = part_result
        return parts

    # ── Build result ────────────────────────────────────────────────────────
    overall_parts = _compute_parts(all_entries)

    per_role_summary: dict = {}
    for role, entries in per_role_entries.items():
        per_role_summary[role] = {
            "total_events": len(entries),
            "parts":        _compute_parts(entries),
        }

    result = {
        "roles_included": sorted(per_role_entries.keys()),
        "total_events":   len(all_entries),
        "parts":          overall_parts,
        "per_role":       per_role_summary,
    }

    # ── Save ────────────────────────────────────────────────────────────────
    suffix   = f"_{run_tag}" if run_tag else ""
    out_path = os.path.join(run_dir, f"latency_comm_overall{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f" Latency & Communication — OVERALL ({len(per_role_entries)} roles)")
    print(f"{'═'*60}")
    for part_name, events in overall_parts.items():
        total = events.get("_total", {})
        model_str = ""
        if total.get("total_model_bytes"):
            model_str = f"  model={total['total_model_bytes']:,} B"
        print(
            f"  [{part_name.upper()}]"
            f"  total_latency={total.get('total_latency_s', 0):.3f} s"
            f"  total_comm={total.get('total_comm_bytes', 0):,} bytes"
            f"{model_str}"
        )
        for event_name, data in events.items():
            if event_name == "_total":
                continue
            l = data["latency_s"]
            model_str = (
                f"  model_total={data['total_model_bytes']:,} B"
                if data.get("total_model_bytes") else ""
            )
            print(
                f"    {event_name:<30} calls={data['call_count']:>4}"
                f"  lat_mean={l['mean']:.4f} s"
                f"  comm_total={data['total_comm_bytes']:,} bytes"
                f"{model_str}"
            )
    print(f"{'═'*60}")
    print(f"[metrics] Overall latency/comm report saved → {out_path}\n")
    return out_path
