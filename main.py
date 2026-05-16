"""
main.py

ThreatCut - Attack-Graph Interdiction Benchmark  (Gurobi, free pip licence)
============================================================================

Instance sized to stay well within the 2000-variable / 2000-constraint
limit of Gurobi's free pip licence on ALL three models, including the
MinBreachPath master problem which grows by ~n_goals variables per iteration.

  Instance parameters
  -------------------
  Nodes  : 20  (+ 1 virtual source = 21 total)
  Levels : 3
  Arcs   : ~44  (ratio ≈ 2.2x nodes, matching paper density)
  Goal nodes : 6   (last level, ~30% of nodes)
  Loss   : Uniform(500, 1500)
  c_atk  : Uniform(10, 30)
  c_def  : Uniform(10, 30)
  B_def  : 40
  B_atk  : 60
  Seed   : 42

Worst-case variable / constraint counts (verified before solving):
  Outer MIP (Models 2 & 3)      :  ~116v /  ~72c   (fixed, per solve)
  Inner u/x/v (Models 2 & 3)    :   ~89v / ~111c   (fixed, per solve)
  Inner MaxBreachD (Model 1)     :  ~314v /  ~355c  (fixed, per solve)
  MinBreachPath master @ 20 iter :  ~228v /  ~641c  (grows with iterations)
  Safe up to ~79 iterations before master hits the 2000-constraint limit.

Install
-------
  pip install gurobipy        # free pip licence included automatically
"""
from __future__ import annotations

import random
import sys
from typing import List

import gurobipy as gp

from model.attack_graph import AttackGraph
from model.paper_algorithm import run_paper_algorithm
from model.model_callbacks import run_callbacks
from model.model_no_callbacks import run_no_callbacks



# Instance generator

def build_instance(
    n_nodes: int = 20,
    n_levels: int = 3,
    pd: float = 0.40,
    ps: float = 0.05,
    seed: int = 42,
) -> AttackGraph:
    """
    Generate a synthetic hierarchical attack graph.

    Topology follows Section 5 of the paper:
      - Node 0        : virtual source (connects to all vulnerability nodes)
      - Level 0       : vulnerability nodes  (no reward)
      - Levels 1..L-2 : transition nodes     (no reward)
      - Level L-1     : goal nodes           (reward ~ Uniform(500, 1500))

    Parameters
    ----------
    n_nodes  : domain nodes, not counting virtual source
    n_levels : number of hierarchy levels
    pd       : probability of a directed arc between adjacent-level node pairs
    ps       : probability of a directed arc between same-level node pairs
    seed     : random seed for reproducibility
    """
    rng = random.Random(seed)
    graph = AttackGraph()

    base = n_nodes // n_levels
    levels: List[List[int]] = []
    start = 1
    for lvl in range(n_levels):
        size = base + (1 if lvl < (n_nodes % n_levels) else 0)
        levels.append(list(range(start, start + size)))
        start += size

    graph.add_node(0, reward=0.0)

    for lvl_idx, level in enumerate(levels):
        is_goal = (lvl_idx == n_levels - 1)
        for nid in level:
            reward = rng.uniform(500, 1500) if is_goal else 0.0
            graph.add_node(nid, reward=reward)

    for v in levels[0]:
        graph.add_arc(0, v, rng.uniform(10, 30), rng.uniform(10, 30))

    for lvl in range(n_levels - 1):
        for tail in levels[lvl]:
            for head in levels[lvl + 1]:
                if rng.random() < pd:
                    graph.add_arc(tail, head, rng.uniform(10, 30), rng.uniform(10, 30))

    for lvl in range(1, n_levels):
        for tail in levels[lvl]:
            for head in levels[lvl]:
                if tail != head and rng.random() < ps:
                    graph.add_arc(tail, head, rng.uniform(10, 30), rng.uniform(10, 30))

    for lvl_idx in range(1, n_levels):
        for nid in levels[lvl_idx]:
            if not any(j == nid for (_, j) in graph.arcs):
                prior = [n for l in levels[:lvl_idx] for n in l]
                tail = rng.choice(prior)
                graph.add_arc(tail, nid, rng.uniform(10, 30), rng.uniform(10, 30))

    return graph



# Variable / constraint safety check

def _check_sizes(graph: AttackGraph, n_levels: int, max_iter: int = 30) -> None:
    """
    Print exact variable/constraint counts for each sub-problem.

    The MinBreachPath master problem grows by (n_goals + n_goals*path_len)
    constraints per iteration; we show the estimate at max_iter iterations.
    Raises SystemExit if any count exceeds 1900 (10% safety margin below 2000).
    """
    arcs = list(graph.arcs.keys())
    n_arcs = len(arcs)
    n_nodes = len(graph.nodes)
    n_goals = sum(1 for nd in graph.nodes.values() if nd.reward > 0)
    non_goal_non_src = n_nodes - n_goals - 1

    w_count = sum(1 for (i, j) in arcs for (l, k) in arcs if j == l)

    # Outer MIP (Models 2 & 3) - fixed size
    outer_v = n_arcs + w_count + 1
    outer_c = 1 + w_count

    # Inner u/x/v (Models 2 & 3) - fixed size (worst case = all arcs free)
    inner_ucb_v = 2 * n_arcs + 1
    inner_ucb_c = 2 + n_nodes + 2 * n_arcs

    # Inner MaxBreachD (Model 1) - fixed size per solve
    inner_d_v = n_arcs + n_goals * n_arcs + n_goals
    inner_d_c = n_goals + n_goals * non_goal_non_src + n_goals * n_arcs + 1

    # MinBreachPath master (Model 1) - grows with iterations
    # Each iteration adds at most n_goals new paths, each path ≈ n_levels arcs
    n_paths_max = max_iter * n_goals
    master_v = n_arcs + n_paths_max + 1              # x + u_p + eta
    master_c = n_paths_max + n_paths_max * n_levels + 1  # obj_cuts + path_arc_links + budget

    limit = 1900  # 5% safety margin below 2000
    items = [
        ("Outer MIP (Models 2 & 3)",        outer_v,    outer_c),
        ("Inner u/x/v (Models 2 & 3)",       inner_ucb_v, inner_ucb_c),
        ("Inner MaxBreachD (Model 1)",        inner_d_v,  inner_d_c),
        (f"MinBreachPath master @{max_iter}i", master_v,  master_c),
    ]

    print(f"\n  Size check  (free-licence limit: 2000, safety threshold: {limit})")
    all_ok = True
    for label, v, c in items:
        worst = max(v, c)
        flag = "OK " if worst < limit else " TOO LARGE"
        print(f"  {label:<36}: {v:4d} vars / {c:4d} constrs  →  {flag}")
        if worst >= limit:
            all_ok = False

    if not all_ok:
        print("\n  ERROR: one or more sub-problems exceed the free-licence limit.")
        print("  Reduce n_nodes or n_levels in build_instance() and retry.")
        sys.exit(1)



# Pretty-print comparison table

def _print_table(results: list) -> None:
    headers = ["Model", "Breach Loss", "Runtime (s)", "Iterations", "Interdicted Arcs"]
    # Compute each column width from the widest value (header or any row cell)
    widths = [
        max(len(str(headers[col])), max(len(str(row[col])) for row in results)) + 2
        for col in range(len(headers))
    ]
    sep  = "+" + "+".join("-" * w for w in widths) + "+"
    head = "|" + "|".join(f" {h:<{w-2}} " for h, w in zip(headers, widths)) + "|"
    print("\n" + sep)
    print(head)
    print(sep)
    for row in results:
        print("|" + "|".join(f" {str(v):<{w-2}} " for v, w in zip(row, widths)) + "|")
    print(sep)



# Main

def main() -> None:
    print("=" * 66)
    print("  ThreatCut - Attack-Graph Interdiction Benchmark (Gurobi)")
    print("  Nandi, Medal, Vadlamani (2016) - COR 75, 118-131")
    print("  Free pip licence  (≤ 2000 vars / 2000 constraints)")
    print("=" * 66)

    try:
        _m = gp.Model()
        _m.dispose()
    except gp.GurobiError as exc:
        print(f"\nERROR: Gurobi not available - {exc}")
        print("Install with:  pip install gurobipy")
        sys.exit(1)

    
    # Build instance  (n=20, 3 levels - safe for all three models)
    N_NODES, N_LEVELS = 20, 3
    print(f"\n[Instance] Building graph ({N_NODES} nodes, {N_LEVELS} levels, seed=42) ...")
    graph = build_instance(n_nodes=N_NODES, n_levels=N_LEVELS, seed=42)
    B_def, B_atk = 40.0, 60.0

    n_goals = sum(1 for nd in graph.nodes.values() if nd.reward > 0)
    trew    = sum(nd.reward for nd in graph.nodes.values())

    print(f"  Nodes (incl. virtual source) : {len(graph.nodes)}")
    print(f"  Arcs                         : {len(graph.arcs)}")
    print(f"  Goal nodes                   : {n_goals}")
    print(f"  Total reward (UB)            : {trew:.1f}")
    print(f"  Defender budget              : {B_def}")
    print(f"  Attacker budget              : {B_atk}")

    _check_sizes(graph, n_levels=N_LEVELS, max_iter=30)

    results = []

    # Model 1 - Paper MINMAX
    print("\n[Model 1] MINMAX exact algorithm (paper, Section 4) ...")
    loss1, idict1, rt1, it1 = run_paper_algorithm(
        graph, B_def, B_atk, epsilon=1e-4, solver_msg=False
    )
    n1 = sum(idict1.values())
    print(f"  Breach Loss={loss1:.2f}  Time={rt1:.2f}s  Iterations={it1}  Interdicted={n1}")
    results.append(["Model 1 - Paper MINMAX (MinBreachPath)", f"{loss1:.2f}", f"{rt1:.2f}", it1, n1])

    # Model 2 - Gurobi lazy callbacks
    print("\n[Model 2] Gurobi lazy-constraint callbacks (new.py) ...")
    loss2, idict2, rt2 = run_callbacks(
        graph, B_def, B_atk, solver_msg=False
    )
    n2 = sum(idict2.values())
    print(f"  Breach Loss={loss2:.2f}  Time={rt2:.2f}s  Interdicted={n2}")
    results.append(["Model 2 - Gurobi Callbacks", f"{loss2:.2f}", f"{rt2:.2f}", "N/A (callbacks)", n2])

    # Model 3 - Iterative Benders
    print("\n[Model 3] Iterative Benders without callbacks ...")
    loss3, idict3, rt3, it3 = run_no_callbacks(
        graph, B_def, B_atk, epsilon=1e-4, solver_msg=False
    )
    n3 = sum(idict3.values())
    print(f"  Breach Loss={loss3:.2f}  Time={rt3:.2f}s  Iterations={it3}  Interdicted={n3}")
    results.append(["Model 3 - Iterative No-Callbacks", f"{loss3:.2f}", f"{rt3:.2f}", it3, n3])

    # Summary
    print("\n" + "=" * 66)
    print("  PERFORMANCE COMPARISON")
    print("=" * 66)
    _print_table(results)

    print("\nColumn descriptions:")
    print("  Breach Loss      : Worst-case attacker reward (lower = better defence)")
    print("  Runtime (s)      : Total wall-clock time")
    print("  Iterations       : Benders outer-loop iterations")
    print("  Interdicted Arcs : Arcs protected by the defender within the budget")
    print()


if __name__ == "__main__":
    main()
