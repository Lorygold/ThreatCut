"""
main.py - Entry point for single-instance runs.

Solves a single attack graph instance with all two solvers:
  1. Benders no-callbacks    - sequential Benders loop (Gurobi)
  2. Benders with callbacks  - lazy constraints via Gurobi callbacks

Usage
-----
    # Default small instance
    python main.py

    # Custom parameters
    python main.py --L 5 --W 10 --d 100 --B_def 75 --B_att 125

    # Paper's "high" cost preset
    python main.py --L 5 --W 10 --d 100 --B_def 75 --B_att 125 --high_costs

    # Draw the graph before solving
    python main.py --draw

    # Show Gurobi solver output
    python main.py --solver_msg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.generator import HIGH_COSTS, LOW_COSTS, draw_attack_graph, generate_attack_graph
from model.new_callbacks import run_new_callbacks
from model.new_no_callbacks import run_new_no_callbacks


def print_results(
    paper_loss: float, paper_time: float, paper_iter: int,
    nocbk_loss: float, nocbk_time: float, nocbk_iter: int,
    cbk_loss:   float, cbk_time:   float,
) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"{'Method':<28} {'Breach Loss':>11} {'Time (s)':>9} {'Iter':>6}")
    print("-" * 60)
    print(f"{'MinMax (paper Algorithm 4.3)':<28} {paper_loss:>11.4f} {paper_time:>9.3f} {paper_iter:>6}")
    print(f"{'Benders no-callbacks':<28} {nocbk_loss:>11.4f} {nocbk_time:>9.3f} {nocbk_iter:>6}")
    print(f"{'Benders with callbacks':<28} {cbk_loss:>11.4f} {cbk_time:>9.3f} {'–':>6}")
    print(f"{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve a single attack-graph interdiction instance with 3 solvers."
    )
    parser.add_argument("--L",    type=int,   default=3,  help="Number of levels (default 3)")
    parser.add_argument("--W",    type=int,   default=3,  help="Nodes per level (default 3)")
    parser.add_argument("--d",    type=int,   default=18, help="Target non-source arc count (default 18)")
    parser.add_argument("--B_def", type=float, default=30, help="Defender budget (default 30)")
    parser.add_argument("--B_att", type=float, default=50, help="Attacker budget (default 50)")
    parser.add_argument("--seed",  type=int,   default=42, help="Random seed (default 42)")
    # Cost/reward ranges (LOW_COSTS preset by default — paper Table 2)
    parser.add_argument("--mina",  type=int, default=LOW_COSTS["mina"], help="Min attack cost")
    parser.add_argument("--maxa",  type=int, default=LOW_COSTS["maxa"], help="Max attack cost")
    parser.add_argument("--mind",  type=int, default=LOW_COSTS["mind"], help="Min interdict cost")
    parser.add_argument("--maxd",  type=int, default=LOW_COSTS["maxd"], help="Max interdict cost")
    parser.add_argument("--minr",  type=int, default=LOW_COSTS["minr"], help="Min goal reward")
    parser.add_argument("--maxr",  type=int, default=LOW_COSTS["maxr"], help="Max goal reward")
    parser.add_argument("--high_costs", action="store_true",
                        help="Use paper's 'high' cost/reward preset (overrides individual cost flags)")
    parser.add_argument("--draw",       action="store_true", help="Draw the attack graph before solving")
    parser.add_argument("--solver_msg", action="store_true", help="Show Gurobi solver output")
    args = parser.parse_args()

    costs = HIGH_COSTS if args.high_costs else dict(
        mina=args.mina, maxa=args.maxa,
        mind=args.mind, maxd=args.maxd,
        minr=args.minr, maxr=args.maxr,
    )

    print(f"\nGenerating attack graph: L={args.L}, W={args.W}, d={args.d}, seed={args.seed}")
    print(f"  Cost/reward : {costs}")
    graph = generate_attack_graph(L=args.L, W=args.W, d=args.d, seed=args.seed, **costs)
    print(graph.summary())
    print(f"  Defender budget : {args.B_def}")
    print(f"  Attacker budget : {args.B_att}")

    if args.draw:
        draw_attack_graph(graph, title=f"Attack Graph (L={args.L}, W={args.W}, d={args.d})")

    print("[2/3] Benders no-callbacks ...")
    nocbk_loss, _, nocbk_iter, nocbk_time = run_new_no_callbacks(
        graph, args.B_def, args.B_att, args.L, args.W,
        solver_msg=args.solver_msg, verbose=True,
    )

    print("[3/3] Benders with callbacks ...")
    cbk_loss, cbk_time = run_new_callbacks(
        graph, args.B_def, args.B_att, args.L, args.W,
        solver_msg=args.solver_msg, verbose=True,
    )

    print_results(
        paper_loss, paper_time, paper_iter,
        nocbk_loss, nocbk_time, nocbk_iter,
        cbk_loss,   cbk_time,
    )


if __name__ == "__main__":
    main()
