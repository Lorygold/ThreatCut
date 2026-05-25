"""
experiments/run_experiments.py

Reproduce the computational experiments from:
    Nandi, Medal, Vadlamani (2016)
    "Interdicting Attack Graphs to Protect Organizations from Cyber Attacks"
    Computers & Operations Research, Vol. 75, pp. 118-131

Three solvers are compared on each instance:
  1. MinMax        — exact algorithm from the paper (Algorithm 4.3, Gurobi)
  2. Benders noCbk — sequential Benders loop, no Gurobi callbacks
  3. Benders Cbk   — lazy constraints via Gurobi callbacks (fastest)

Graph parameters follow Table 2 of the paper:
  - Nodes  : 50, 100, 150, 200  (≈ 1 + L × W)
  - Arcs   : ≈ 2.15 × nodes     (d ≈ 2.15 × nodes − W source arcs)
  - Costs/rewards: "low" preset by default (paper Table 2, low parameter levels)

Usage
-----
    python experiments/run_experiments.py               # full experiment
    python experiments/run_experiments.py --quick       # small subset only
    python experiments/run_experiments.py --csv out.csv # save to CSV
    python experiments/run_experiments.py --high_costs  # paper's "high" cost preset
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generator import HIGH_COSTS, LOW_COSTS, generate_attack_graph
from model.new_callbacks import run_new_callbacks
from model.new_no_callbacks import run_new_no_callbacks



# Parameter grid  (Table 2 of the paper)
# Each entry: (L, W, d, B_defender, B_attacker)
#   nodes ≈ 1 + L*W,   d ≈ round(2.15*(1+L*W)) − W

FULL_GRID = [
    # (L,  W,   d,   B_def, B_att)
    # ~50 nodes
    (5,  10,  100,   75,  125),
    (7,   7,  100,   75,  125),
    # ~100 nodes 
    (5,  20,  197,  150,  150),
    (5,  20,  197,  250,  300),
    (2,  50,  167,  150,  150),
    # ~150 nodes 
    (5,  30,  295,  275,  325),
    # ~200 nodes 
    (5,  40,  393,  375,  425),
]

QUICK_GRID = [
    (5,  10,  100,   75,  125),
    (7,   7,  100,   75,  125),
    (5,  20,  197,  150,  150),
]

N_INSTANCES = 10



# Single-instance runner

def run_one(
    L: int, W: int, d: int,
    B_defender: float, B_attacker: float,
    instance_seed: int,
    costs: dict,
) -> dict:
    """Run all three solvers on a single graph instance."""
    graph = generate_attack_graph(L=L, W=W, d=d, seed=instance_seed, **costs)

    row: dict = {
        "L": L, "W": W, "d": d,
        "B_def": B_defender, "B_att": B_attacker,
        "seed": instance_seed,
        "n_nodes": len(graph.nodes),
        "n_arcs":  len(graph.arcs),
    }

    # 1. Benders no-callbacks
    nc_loss, _, nc_iter, nc_time = run_new_no_callbacks(
        graph, B_defender, B_attacker, L, W, verbose=False
    )
    row["nc_loss"]   = round(nc_loss, 4)
    row["nc_iter"]   = nc_iter
    row["nc_time_s"] = round(nc_time, 3)

    # 2. Benders with callbacks
    cb_loss, cb_time = run_new_callbacks(
        graph, B_defender, B_attacker, L, W, verbose=False
    )
    row["cb_loss"]   = round(cb_loss, 4)
    row["cb_time_s"] = round(cb_time, 3)

    return row



# Aggregation helpers

def _avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"



# Main experiment loop

def run_experiments(grid: list, n_instances: int, csv_path: str | None,
                    costs: dict) -> None:
    all_rows: list[dict] = []

    header = (
        f"{'L':>2} {'W':>3} {'d':>4} {'Bd':>5} {'Ba':>5} | "
        f"{'MinMax loss':>11} {'iter':>4} {'time':>6} | "
        f"{'NoCbk loss':>10} {'iter':>4} {'time':>6} | "
        f"{'Cbk loss':>8} {'time':>6}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for (L, W, d, B_def, B_att) in grid:
        combo_rows = []

        for seed in range(n_instances):
            print(f"  Running L={L} W={W} d={d} Bd={B_def} Ba={B_att} "
                  f"seed={seed} ...", end="\r")
            row = run_one(L, W, d, B_def, B_att, instance_seed=seed, costs=costs)
            combo_rows.append(row)
            all_rows.append(row)

        mm_loss = _avg(combo_rows, "mm_loss")
        mm_iter = _avg(combo_rows, "mm_iter")
        mm_time = _avg(combo_rows, "mm_time_s")
        nc_loss = _avg(combo_rows, "nc_loss")
        nc_iter = _avg(combo_rows, "nc_iter")
        nc_time = _avg(combo_rows, "nc_time_s")
        cb_loss = _avg(combo_rows, "cb_loss")
        cb_time = _avg(combo_rows, "cb_time_s")

        print(
            f"{L:>2} {W:>3} {d:>4} {B_def:>5} {B_att:>5} | "
            f"{_fmt(mm_loss):>11} {_fmt(mm_iter,1):>4} {_fmt(mm_time,3):>6} | "
            f"{_fmt(nc_loss):>10} {_fmt(nc_iter,1):>4} {_fmt(nc_time,3):>6} | "
            f"{_fmt(cb_loss):>8} {_fmt(cb_time,3):>6}"
        )

    print(sep)
    print(f"\nTotal instances run: {len(all_rows)}")

    if csv_path and all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Results saved to {csv_path}")



# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run attack graph interdiction experiments (3 solvers)."
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run only a small subset of the parameter grid.")
    parser.add_argument("--csv", metavar="PATH",
                        help="Save all results to a CSV file at PATH.")
    parser.add_argument("--instances", type=int, default=N_INSTANCES,
                        help=f"Instances per parameter combination (default {N_INSTANCES}).")
    parser.add_argument("--high_costs", action="store_true",
                        help="Use paper's 'high' cost/reward preset instead of 'low'.")
    args = parser.parse_args()

    grid  = QUICK_GRID if args.quick else FULL_GRID
    costs = HIGH_COSTS if args.high_costs else LOW_COSTS
    run_experiments(grid=grid, n_instances=args.instances, csv_path=args.csv, costs=costs)
