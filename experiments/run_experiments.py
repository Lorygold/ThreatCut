"""
experiments/run_experiments.py

Reproduce the computational experiments from:
    Nandi, Medal, Vadlamani (2016)
    "Interdicting Attack Graphs to Protect Organizations from Cyber Attacks"
    Computers & Operations Research, Vol. 75, pp. 118-131

For each (L, W, d, B_defender, B_attacker) combination:
  - Generate N_INSTANCES random attack graphs
  - Solve with the exact CCG algorithm
  - Solve with the LP heuristic
  - Solve with the greedy heuristic
  - Report average breach loss, iterations, solve time, and optimality gap

Results are printed as a formatted table and optionally saved to CSV.

Usage
-----
    python experiments/run_experiments.py               # full experiment
    python experiments/run_experiments.py --quick       # small subset only
    python experiments/run_experiments.py --csv out.csv # save to CSV
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generator import generate_instances
from model.algorithm import CCGResult, run_ccg_algorithm, run_heuristic_greedy, run_heuristic_lp


# Experiment configuration

# Full parameter grid from the paper (Section 5)
FULL_GRID = [
    # (L, W, d, B_defender, B_attacker)
    (3, 3, 2, 10, 15),
    (3, 3, 2, 15, 15),
    (3, 3, 3, 10, 20),
    (4, 3, 2, 15, 20),
    (4, 4, 2, 20, 25),
    (4, 4, 3, 20, 25),
    (5, 3, 2, 20, 30),
    (5, 4, 2, 25, 30),
    (5, 5, 2, 30, 35),
    (5, 5, 3, 30, 35),
]

# Smaller grid for quick smoke test
QUICK_GRID = [
    (3, 3, 2, 10, 15),
    (3, 3, 3, 10, 20),
    (4, 3, 2, 15, 20),
]

N_INSTANCES = 10   # instances per parameter combination (paper uses 10)


# Single-instance runner

def run_one(
    L: int, W: int, d: int,
    B_defender: float, B_attacker: float,
    instance_seed: int,
) -> dict:
    """Run all three methods on a single graph instance."""
    from data.generator import generate_attack_graph

    graph = generate_attack_graph(L=L, W=W, d=d, seed=instance_seed)

    row: dict = {
        "L": L, "W": W, "d": d,
        "B_def": B_defender, "B_att": B_attacker,
        "seed": instance_seed,
        "n_nodes": len(graph.nodes),
        "n_arcs":  len(graph.arcs),
    }

    # --- Exact CCG ---
    res_ccg = run_ccg_algorithm(graph, B_defender, B_attacker)
    row["ccg_loss"]   = round(res_ccg.breach_loss, 4)
    row["ccg_iter"]   = res_ccg.n_iterations
    row["ccg_time_s"] = round(res_ccg.solve_time_s, 3)
    row["ccg_gap"]    = round(res_ccg.optimality_gap, 6)

    # --- LP heuristic ---
    res_lp = run_heuristic_lp(graph, B_defender, B_attacker)
    row["lp_loss"]   = round(res_lp.breach_loss, 4)
    row["lp_iter"]   = res_lp.n_iterations
    row["lp_time_s"] = round(res_lp.solve_time_s, 3)
    row["lp_gap"]    = round(res_lp.optimality_gap, 6)

    # --- Greedy heuristic ---
    res_gr = run_heuristic_greedy(graph, B_defender, B_attacker)
    row["gr_loss"]   = round(res_gr.breach_loss, 4)
    row["gr_iter"]   = res_gr.n_iterations
    row["gr_time_s"] = round(res_gr.solve_time_s, 3)
    row["gr_gap"]    = round(res_gr.optimality_gap, 6)

    return row


# Aggregation helpers

def _avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


# Main experiment loop

def run_experiments(grid: list, n_instances: int, csv_path: str | None) -> None:
    all_rows: list[dict] = []

    # Table header
    header = (
        f"{'L':>2} {'W':>2} {'d':>2} {'Bd':>4} {'Ba':>4} | "
        f"{'CCG loss':>9} {'iter':>4} {'time':>6} | "
        f"{'LP loss':>9} {'gap%':>6} {'time':>6} | "
        f"{'GR loss':>9} {'gap%':>6} {'time':>6}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for (L, W, d, B_def, B_att) in grid:
        combo_rows = []

        for seed in range(n_instances):
            print(
                f"  Running L={L} W={W} d={d} Bd={B_def} Ba={B_att} "
                f"seed={seed} ...",
                end="\r",
            )
            row = run_one(L, W, d, B_def, B_att, instance_seed=seed)
            combo_rows.append(row)
            all_rows.append(row)

        # Print averaged row
        ccg_loss = _avg(combo_rows, "ccg_loss")
        ccg_iter = _avg(combo_rows, "ccg_iter")
        ccg_time = _avg(combo_rows, "ccg_time_s")
        lp_loss  = _avg(combo_rows, "lp_loss")
        lp_gap   = _avg(combo_rows, "lp_gap") * 100
        lp_time  = _avg(combo_rows, "lp_time_s")
        gr_loss  = _avg(combo_rows, "gr_loss")
        gr_gap   = _avg(combo_rows, "gr_gap") * 100
        gr_time  = _avg(combo_rows, "gr_time_s")

        print(
            f"{L:>2} {W:>2} {d:>2} {B_def:>4} {B_att:>4} | "
            f"{_fmt(ccg_loss):>9} {_fmt(ccg_iter,1):>4} {_fmt(ccg_time,3):>6} | "
            f"{_fmt(lp_loss):>9} {_fmt(lp_gap,2):>5}% {_fmt(lp_time,3):>6} | "
            f"{_fmt(gr_loss):>9} {_fmt(gr_gap,2):>5}% {_fmt(gr_time,3):>6}"
        )

    print(sep)
    print(f"\nTotal instances run: {len(all_rows)}")

    # Optionally save to CSV
    if csv_path and all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Results saved to {csv_path}")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attack graph interdiction experiments.")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only a small subset of the parameter grid (faster)."
    )
    parser.add_argument(
        "--csv", metavar="PATH",
        help="Save all results to a CSV file at PATH."
    )
    parser.add_argument(
        "--instances", type=int, default=N_INSTANCES,
        help=f"Number of random instances per parameter combination (default {N_INSTANCES})."
    )
    args = parser.parse_args()

    grid = QUICK_GRID if args.quick else FULL_GRID
    run_experiments(grid=grid, n_instances=args.instances, csv_path=args.csv)
