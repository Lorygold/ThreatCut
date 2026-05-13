"""
data/generator.py

Synthetic attack graph generator matching EXACTLY the procedure described in
Section 5 of:

    Nandi, Medal, Vadlamani (2016)
    "Interdicting Attack Graphs to Protect Organizations from Cyber Attacks"
    Computers & Operations Research, Vol. 75, pp. 118-131

Generation procedure (Section 5, paragraphs 3-4):
  1. n_nodes nodes split evenly across r levels.
     Level 0 = vulnerability nodes; level r-1 = goal nodes; rest = transition.
  2. Arc from level l to l+1 with probability pd for each pair of nodes.
  3. Arc between nodes in the SAME level with probability ps.
  4. Each transition/goal node with no incoming arc receives one random
     incoming arc from a prior level (ensures full connectivity).

Parameter values from Table 2 of the paper:
  Network sizes       : 50, 100, 150, 200 nodes
  (size, levels)      : (50,5),(50,7),(50,10),(100,2),(100,5),(150,5),(200,5)
  Target arc density  : ≈ 2.15 × n_nodes
  Loss (reward)       : Uniform(500,1500) or Uniform(1000,2000)
  Attack cost c_a     : Uniform(10,30)    or Uniform(30,50)
  Defense cost c_d    : Uniform(10,30)    or Uniform(30,50)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.attack_graph import AttackGraph



# Core generator

def generate_attack_graph(
    n_nodes: int,
    n_levels: int,
    pd: float,
    ps: float = 0.0,
    reward_low: int = 500,
    reward_high: int = 1500,
    attack_cost_low: int = 10,
    attack_cost_high: int = 30,
    defend_cost_low: int = 10,
    defend_cost_high: int = 30,
    seed: Optional[int] = None,
) -> AttackGraph:
    """
    Generate a synthetic attack graph following the paper's exact procedure.

    Parameters
    ----------
    n_nodes          : Total number of nodes.
    n_levels         : Number of levels r >= 2.
    pd               : Probability of arc between any pair in adjacent levels.
    ps               : Probability of arc between any pair in the same level.
    reward_low/high  : Uniform range for goal node rewards.
    attack_cost_low/high  : Uniform range for c_a (attacker arc cost).
    defend_cost_low/high  : Uniform range for c_d (defender arc cost).
    seed             : RNG seed for reproducibility.
    """
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2")
    if n_nodes < n_levels:
        raise ValueError("n_nodes must be >= n_levels")

    rng = random.Random(seed)
    graph = AttackGraph()

    # Step 1: assign nodes to levels
    base, rem = divmod(n_nodes, n_levels)
    level_nodes: List[List[int]] = []
    nid = 0
    for lv in range(n_levels):
        size = base + (1 if lv < rem else 0)
        ids = list(range(nid, nid + size))
        level_nodes.append(ids)
        is_goal = (lv == n_levels - 1)
        for node_id in ids:
            reward = float(rng.randint(reward_low, reward_high)) if is_goal else 0.0
            graph.add_node(node_id=node_id, level=lv, reward=reward)
        nid += size

    # Step 2: inter-level arcs with probability pd
    for lv in range(n_levels - 1):
        for src in level_nodes[lv]:
            for dst in level_nodes[lv + 1]:
                if rng.random() < pd:
                    ca = float(rng.randint(attack_cost_low, attack_cost_high))
                    cd = float(rng.randint(defend_cost_low, defend_cost_high))
                    graph.add_arc(src=src, dst=dst, cost_attack=ca, cost_interdict=cd)

    # Step 3: same-level arcs with probability ps
    if ps > 0.0:
        for lv in range(n_levels):
            nodes = level_nodes[lv]
            for i, src in enumerate(nodes):
                for dst in nodes[i + 1:]:
                    if rng.random() < ps:
                        ca = float(rng.randint(attack_cost_low, attack_cost_high))
                        cd = float(rng.randint(defend_cost_low, defend_cost_high))
                        graph.add_arc(src=src, dst=dst, cost_attack=ca, cost_interdict=cd)

    # Step 4: guarantee every transition/goal node has >= 1 in-arc
    for lv in range(1, n_levels):
        for node_id in level_nodes[lv]:
            has_in = any(d == node_id for (_, d) in graph.arcs)
            if not has_in:
                prior = [n for prev in range(lv) for n in level_nodes[prev]]
                src = rng.choice(prior)
                ca = float(rng.randint(attack_cost_low, attack_cost_high))
                cd = float(rng.randint(defend_cost_low, defend_cost_high))
                graph.add_arc(src=src, dst=node_id, cost_attack=ca, cost_interdict=cd)

    return graph


# Paper-calibrated wrapper: auto-computes pd to get ~2.15 * n_nodes arcs

def generate_paper_instance(
    n_nodes: int,
    n_levels: int,
    seed: Optional[int] = None,
    reward_range: Tuple[int, int] = (500, 1500),
    attack_cost_range: Tuple[int, int] = (10, 30),
    defend_cost_range: Tuple[int, int] = (10, 30),
) -> AttackGraph:
    """
    Generate a paper-calibrated instance with ≈ 2.15 * n_nodes arcs.

    pd is computed so that the expected number of inter-level arcs equals
    the target:  target = 2.15 * n_nodes - expected_step4_arcs
    Step 4 adds at most one arc per unreachable node (expected ~0 for pd>0.5).
    """
    base, rem = divmod(n_nodes, n_levels)
    sizes = [base + (1 if i < rem else 0) for i in range(n_levels)]
    denom = sum(sizes[i] * sizes[i + 1] for i in range(n_levels - 1))
    target = 2.15 * n_nodes
    pd = min(target / denom if denom > 0 else 1.0, 1.0)

    return generate_attack_graph(
        n_nodes=n_nodes,
        n_levels=n_levels,
        pd=pd,
        ps=0.0,
        reward_low=reward_range[0],
        reward_high=reward_range[1],
        attack_cost_low=attack_cost_range[0],
        attack_cost_high=attack_cost_range[1],
        defend_cost_low=defend_cost_range[0],
        defend_cost_high=defend_cost_range[1],
        seed=seed,
    )


def generate_instances(
    n_nodes: int,
    n_levels: int,
    n_instances: int = 4,
    base_seed: int = 0,
    **kwargs,
) -> List[AttackGraph]:
    """
    Generate n_instances independent graphs (paper uses 4 per combination).
    """
    return [
        generate_paper_instance(
            n_nodes=n_nodes,
            n_levels=n_levels,
            seed=base_seed + i,
            **kwargs,
        )
        for i in range(n_instances)
    ]


# Visualization

def draw_attack_graph(
    graph: AttackGraph,
    title: str = "Attack Graph",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    g = graph._graph
    levels: dict = {}
    for node_id, node in graph.nodes.items():
        levels.setdefault(node.level, []).append(node_id)

    pos = {}
    for lv, nids in sorted(levels.items()):
        n = len(nids)
        for rank, nid in enumerate(sorted(nids)):
            pos[nid] = (float(lv), float(rank) - (n - 1) / 2.0)

    goal_ids = {n.id for n in graph.goal_nodes}
    colors = []
    for nid in g.nodes():
        lv = graph.nodes[nid].level
        if lv == 0:
            colors.append("#90EE90")
        elif nid in goal_ids:
            colors.append("#FF6B6B")
        else:
            colors.append("#AED6F1")

    node_labels = {
        nid: (f"{nid}\nr={graph.nodes[nid].reward:.0f}"
              if graph.nodes[nid].reward > 0 else str(nid))
        for nid in g.nodes()
    }
    edge_labels = {
        (a.src, a.dst): f"a={a.cost_attack:.0f}\nd={a.cost_interdict:.0f}"
        for a in graph.arcs.values()
    }

    fig, ax = plt.subplots(figsize=(max(8, graph.num_levels * 3), 7))
    ax.set_title(title, fontsize=12, fontweight="bold")
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=700, ax=ax)
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=7, ax=ax)
    nx.draw_networkx_edges(g, pos, arrows=True, arrowsize=16,
                           edge_color="#444", ax=ax,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                 font_size=6, ax=ax)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#90EE90", label="Vulnerability (level 0)"),
        Patch(facecolor="#AED6F1", label="Transition"),
        Patch(facecolor="#FF6B6B", label="Goal (last level)"),
    ], loc="upper left", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# Quick smoke test - run this file directly to verify the generator works                                                                                                                                                     ─╯

if __name__ == "__main__":
    for (n, lv) in [(50, 5), (50, 7), (100, 5), (100, 2)]:
        g = generate_paper_instance(n_nodes=n, n_levels=lv, seed=0)
        ratio = len(g.arcs) / n
        print(f"  nodes={n:3d}  levels={lv}  arcs={len(g.arcs):4d}  "
              f"ratio={ratio:.2f}  (target≈2.15)")
    print("\nDrawing 50-node 5-level graph...")
    g = generate_paper_instance(n_nodes=50, n_levels=5, seed=0)
    print(g.summary())
    draw_attack_graph(g, title="Attack Graph — 50 nodes, 5 levels")