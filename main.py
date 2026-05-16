"""
main.py

Entry point for the attack-graph interdiction benchmark.

Generates the same attack-graph instance described in Section 5 of:
  Nandi, Medal, Vadlamani (2016) - Computers & Operations Research 75, 118-131

Then runs the three models and prints a comparative performance table.

Instance parameters (low-level settings, Table 2 in the paper)
---------------------------------------------------------------
  Nodes  : 50
  Levels : 5
  Arcs   : ~2.15 × nodes  (≈ 107)
  Loss   : Uniform(500, 1500)
  c_atk  : Uniform(10, 30)
  c_def  : Uniform(10, 30)
  B_def  : 75
  B_atk  : 125
  Random seed: 42 (for full reproducibility)
"""
from __future__ import annotations

import random
import time
from typing import List

from model.attack_graph import AttackGraph
from model.paper_algorithm import run_paper_algorithm
from model.model_callbacks import run_callbacks
from model.model_no_callbacks import run_no_callbacks



# Instance generator


def build_paper_instance(
    n_nodes: int = 50,
    n_levels: int = 5,
    pd: float = 0.25,   # inter-level arc probability (tuned for ~2.15x arcs)
    ps: float = 0.03,   # intra-level arc probability
    seed: int = 42,
) -> AttackGraph:
    """
    Generate a synthetic attack graph matching the paper's experimental setup.

    The graph uses a hierarchical topology:
      - Level 0         : vulnerability (source) nodes  → all connect to node 0
      - Levels 1..L-2   : transition nodes
      - Level L-1       : goal nodes (non-zero reward)

    A virtual source node 0 is added and connected to all vulnerability nodes.

    Parameters
    ----------
    n_nodes  : total number of domain nodes (excluding virtual source)
    n_levels : number of levels in the hierarchy
    pd       : probability of an arc between adjacent levels
    ps       : probability of an arc within the same level
    seed     : random seed

    Returns
    -------
    graph : populated AttackGraph
    """
    rng = random.Random(seed)

    graph = AttackGraph()

    # Divide n_nodes into levels as evenly as possible
    base = n_nodes // n_levels
    remainder = n_nodes % n_levels
    level_sizes: List[int] = []
    start = 1  # node IDs start at 1; 0 is the virtual source
    levels: List[List[int]] = []

    for lvl in range(n_levels):
        size = base + (1 if lvl < remainder else 0)
        node_ids = list(range(start, start + size))
        levels.append(node_ids)
        start += size

    vulnerability_nodes = levels[0]
    goal_nodes = levels[-1]
    transition_nodes = [n for lvl in levels[1:-1] for n in lvl]

    # Add virtual source node 0 (no reward, no cost)
    graph.add_node(0, reward=0.0)

    # Add domain nodes
    for lvl_idx, level in enumerate(levels):
        for node_id in level:
            if lvl_idx == n_levels - 1:
                # Goal node: assign reward ~ Uniform(500, 1500)
                reward = rng.uniform(500, 1500)
            else:
                reward = 0.0
            graph.add_node(node_id, reward=reward)

    # Virtual source arcs: 0 -> each vulnerability node
    for v in vulnerability_nodes:
        c_atk = rng.uniform(10, 30)
        c_def = rng.uniform(10, 30)
        graph.add_arc(0, v, cost_attack=c_atk, cost_interdict=c_def)

    # Inter-level arcs (level l -> level l+1)
    for lvl in range(n_levels - 1):
        for tail in levels[lvl]:
            for head in levels[lvl + 1]:
                if rng.random() < pd:
                    c_atk = rng.uniform(10, 30)
                    c_def = rng.uniform(10, 30)
                    graph.add_arc(tail, head, cost_attack=c_atk, cost_interdict=c_def)

    # Intra-level arcs (within same level, only transition/goal)
    for lvl in range(1, n_levels):
        for i, tail in enumerate(levels[lvl]):
            for head in levels[lvl]:
                if tail != head and rng.random() < ps:
                    c_atk = rng.uniform(10, 30)
                    c_def = rng.uniform(10, 30)
                    graph.add_arc(tail, head, cost_attack=c_atk, cost_interdict=c_def)

    # Guarantee every transition/goal node has at least one incoming arc
    all_domain = [n for level in levels for n in level]
    for lvl_idx in range(1, n_levels):
        for node_id in levels[lvl_idx]:
            has_incoming = any(j == node_id for (_, j) in graph.arcs)
            if not has_incoming:
                # Pick a random node from any prior level
                prior_nodes = [n for lvl in levels[:lvl_idx] for n in lvl]
                tail = rng.choice(prior_nodes)
                c_atk = rng.uniform(10, 30)
                c_def = rng.uniform(10, 30)
                graph.add_arc(tail, node_id, cost_attack=c_atk, cost_interdict=c_def)

    return graph



# Pretty-print table


def _print_table(results: list) -> None:
    """Print a formatted comparison table of the three models."""
    col_names = [
        "Model",
        "Breach Loss",
        "Runtime (s)",
        "Iterations",
        "Interdicted Arcs",
    ]
    col_widths = [30, 14, 14, 12, 18]

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    header = "|" + "|".join(
        f" {name:<{w-2}} " for name, w in zip(col_names, col_widths)
    ) + "|"

    print("\n" + sep)
    print(header)
    print(sep)
    for row in results:
        line = "|" + "|".join(
            f" {str(val):<{w-2}} " for val, w in zip(row, col_widths)
        ) + "|"
        print(line)
    print(sep)



# Main


def main() -> None:
    print("=" * 60)
    print("  Attack-Graph Interdiction Benchmark")
    print("  Nandi, Medal, Vadlamani (2016) - COR 75, 118-131")
    print("=" * 60)

    # Build instance
    print("\n[Instance] Building graph (50 nodes, 5 levels, seed=42) ...")
    graph = build_paper_instance(
        n_nodes=50,
        n_levels=5,
        seed=42,
    )
    n_arcs = len(graph.arcs)
    n_nodes = len(graph.nodes)
    B_def = 75.0
    B_atk = 125.0

    print(f"  Nodes (incl. virtual source) : {n_nodes}")
    print(f"  Arcs                         : {n_arcs}")
    print(f"  Defender budget              : {B_def}")
    print(f"  Attacker budget              : {B_atk}")

    results = []

    
    # Model 1 - Paper MINMAX (MINBREACHPATH + MAXBREACH)
    
    print("\n[Model 1] Running MINMAX algorithm (paper exact) ...")
    t_start = time.time()
    loss1, interdict1, rt1, iters1 = run_paper_algorithm(
        graph, B_def, B_atk, epsilon=1e-4, solver_msg=False
    )
    interdicted1 = sum(1 for v in interdict1.values() if v == 1)
    print(f"  Done. Loss={loss1:.2f}  Time={rt1:.2f}s  Iters={iters1}")
    results.append([
        "Model 1 - Paper MINMAX",
        f"{loss1:.2f}",
        f"{rt1:.2f}",
        iters1,
        interdicted1,
    ])

    
    # Model 2 - Gurobi Callbacks (refactored new.py)
    
    print("\n[Model 2] Running Gurobi callbacks model (new.py) ...")
    loss2, interdict2, rt2 = run_callbacks(
        graph, B_def, B_atk, solver_msg=False
    )
    interdicted2 = sum(1 for v in interdict2.values() if v == 1)
    print(f"  Done. Loss={loss2:.2f}  Time={rt2:.2f}s")
    results.append([
        "Model 2 - Gurobi Callbacks",
        f"{loss2:.2f}",
        f"{rt2:.2f}",
        "N/A (1 outer solve)",
        interdicted2,
    ])

    
    # Model 3 - No Callbacks iterative (refactored new_Gurobi_no_callbacks.py)
    
    print("\n[Model 3] Running iterative no-callbacks model ...")
    loss3, interdict3, rt3, iters3 = run_no_callbacks(
        graph, B_def, B_atk, epsilon=1e-4, solver_msg=False
    )
    interdicted3 = sum(1 for v in interdict3.values() if v == 1)
    print(f"  Done. Loss={loss3:.2f}  Time={rt3:.2f}s  Iters={iters3}")
    results.append([
        "Model 3 - Iterative No-Callbacks",
        f"{loss3:.2f}",
        f"{rt3:.2f}",
        iters3,
        interdicted3,
    ])

    
    # Print summary table
    
    print("\n" + "=" * 60)
    print("  PERFORMANCE COMPARISON")
    print("=" * 60)
    _print_table(results)

    print("\nColumn descriptions:")
    print("  Breach Loss      : Optimal worst-case attacker reward (lower = better defence)")
    print("  Runtime (s)      : Total wall-clock time in seconds")
    print("  Iterations       : Number of master/sub-problem iterations (outer loop cycles)")
    print("  Interdicted Arcs : Number of arcs chosen for protection by the defender")
    print()


if __name__ == "__main__":
    main()
