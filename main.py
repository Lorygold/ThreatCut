"""
main.py - Entry point for single-instance runs and convergence plots.

Solves a single attack graph instance with the exact CCG algorithm and the
two heuristics, prints a comparison table, and shows a convergence plot.

Usage
-----
    # Default small instance
    python main.py

    # Custom parameters
    python main.py --L 4 --W 4 --d 3 --B_def 20 --B_att 25 --seed 42

    # Save convergence plot to file
    python main.py --save_plot convergence.png

    # Validate against small exact bilevel (only for small graphs)
    python main.py --L 3 --W 3 --d 2 --validate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.generator import draw_attack_graph, generate_attack_graph
from model.algorithm import run_ccg_algorithm, run_heuristic_greedy, run_heuristic_lp
from model.bilevel import solve_bilevel_small
from model.new_callbacks import run_new_callbacks
from model.new_no_callbacks import run_new_no_callbacks


# Convergence plot

def plot_convergence(
    lower_bounds: list[float],
    upper_bounds: list[float],
    title: str = "CCG Convergence",
    save_path: str | None = None,
) -> None:
    """
    Plot lower and upper bounds vs. iteration number.

    This reproduces Figure 3 of the paper: the two curves start apart
    and converge to the optimal value as new attack paths are added.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping convergence plot.")
        return

    iterations = list(range(1, len(lower_bounds) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, lower_bounds, "b-o", label="Lower bound (master)", linewidth=2)
    ax.plot(iterations, upper_bounds, "r-s", label="Upper bound (subproblem)", linewidth=2)

    if lower_bounds and upper_bounds:
        opt = upper_bounds[-1]
        ax.axhline(y=opt, color="gray", linestyle="--", linewidth=1, label=f"Optimal = {opt:.2f}")

    ax.set_xlabel("CCG Iteration", fontsize=12)
    ax.set_ylabel("Breach Loss", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Convergence plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# Comparison table printer

def print_comparison(
    ccg_loss: float, ccg_iter: int, ccg_time: float,
    lp_loss:  float, lp_gap:   float, lp_time:  float,
    gr_loss:  float, gr_gap:   float, gr_time:  float,
    exact_loss: float | None = None,
) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"{'Method':<22} {'Breach Loss':>11} {'Gap%':>7} {'Time (s)':>9} {'Iter':>5}")
    print("-" * 62)
    print(f"{'Exact CCG':<22} {ccg_loss:>11.4f} {'-':>7} {ccg_time:>9.3f} {ccg_iter:>5}")
    print(f"{'LP heuristic':<22} {lp_loss:>11.4f} {lp_gap*100:>6.2f}% {lp_time:>9.3f} {'-':>5}")
    print(f"{'Greedy heuristic':<22} {gr_loss:>11.4f} {gr_gap*100:>6.2f}% {gr_time:>9.3f} {'-':>5}")
    if exact_loss is not None:
        print("-" * 62)
        print(f"{'Bilevel MIP (exact)':<22} {exact_loss:>11.4f} {'-':>7} {'-':>9} {'-':>5}")
    print(f"{sep}\n")


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve a single attack-graph interdiction instance."
    )
    parser.add_argument("--L",      type=int,   default=3,  help="Number of levels (default 3)")
    parser.add_argument("--W",      type=int,   default=3,  help="Nodes per level (default 3)")
    parser.add_argument("--d",      type=int,   default=2,  help="Out-degree per node (default 2)")
    parser.add_argument("--B_def",  type=float, default=10, help="Defender budget (default 10)")
    parser.add_argument("--B_att",  type=float, default=15, help="Attacker budget (default 15)")
    parser.add_argument("--seed",   type=int,   default=42, help="Random seed (default 42)")
    parser.add_argument("--validate", action="store_true",
                        help="Also solve with small bilevel MIP for validation.")
    parser.add_argument("--draw",   action="store_true",
                        help="Draw the attack graph before solving.")
    parser.add_argument("--save_plot", metavar="PATH",
                        help="Save convergence plot to file instead of showing it.")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip convergence plot entirely.")
    parser.add_argument("--solver_msg", action="store_true",
                        help="Show Gurobi solver output.")
    args = parser.parse_args()

    # Generate instance
    print(f"\nGenerating attack graph: L={args.L}, W={args.W}, d={args.d}, seed={args.seed}")
    graph = generate_attack_graph(L=args.L, W=args.W, d=args.d, seed=args.seed)
    print(graph.summary())
    print(f"  Defender budget : {args.B_def}")
    print(f"  Attacker budget : {args.B_att}")
    print(f"  All paths       : {len(graph.get_all_paths())}")

    if args.draw:
        draw_attack_graph(
            graph,
            title=f"Attack Graph (L={args.L}, W={args.W}, d={args.d})",
        )

    # Gurobi models
    run_new_callbacks(graph, args.B_def, args.B_att, args.L, args.W, solver_msg=args.solver_msg)
    run_new_no_callbacks(graph, args.B_def, args.B_att, args.L, args.W, solver_msg=args.solver_msg)

    # Solve with all three CCG methods
    print("\nSolving with Exact CCG ...")
    res_ccg = run_ccg_algorithm(graph, args.B_def, args.B_att)

    print("Solving with LP heuristic ...")
    res_lp  = run_heuristic_lp(graph, args.B_def, args.B_att)

    print("Solving with Greedy heuristic ...")
    res_gr  = run_heuristic_greedy(graph, args.B_def, args.B_att)

    # Optional: validate with small bilevel MIP
    exact_loss = None
    if args.validate:
        print("Solving with bilevel MIP (validation) ...")
        exact_loss, _, n_paths = solve_bilevel_small(graph, args.B_def, args.B_att)
        print(f"  Bilevel MIP: breach_loss={exact_loss:.4f}, paths enumerated={n_paths}")

    # Print comparison table
    print_comparison(
        ccg_loss=res_ccg.breach_loss,
        ccg_iter=res_ccg.n_iterations,
        ccg_time=res_ccg.solve_time_s,
        lp_loss=res_lp.breach_loss,
        lp_gap=res_lp.optimality_gap,
        lp_time=res_lp.solve_time_s,
        gr_loss=res_gr.breach_loss,
        gr_gap=res_gr.optimality_gap,
        gr_time=res_gr.solve_time_s,
        exact_loss=exact_loss,
    )

    # Print optimal interdiction plan
    interdicted = [arc for arc, v in res_ccg.x_optimal.items() if v == 1]
    print(f"Optimal interdiction plan (CCG): {interdicted}")
    print(f"CCG converged in {res_ccg.n_iterations} iteration(s)\n")

    # Convergence plot
    if not args.no_plot:
        plot_convergence(
            res_ccg.lower_bounds,
            res_ccg.upper_bounds,
            title=f"CCG Convergence  (L={args.L}, W={args.W}, d={args.d})",
            save_path=args.save_plot,
        )


if __name__ == "__main__":
    main()
