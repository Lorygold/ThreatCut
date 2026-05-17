"""
data/generator.py

Synthetic attack graph generator following the experimental setup of:
    Nandi, Medal, Vadlamani (2016)
    "Interdicting Attack Graphs to Protect Organizations from Cyber Attacks"
    Computers & Operations Research, Vol. 75, pp. 118-131

Graph structure (Section 5 - Computational Experiments):
  - L levels (excluding the source at level 0)
  - W nodes per level
  - Each node at level l is connected to exactly d nodes chosen
    uniformly at random from level l+1
  - arc cost_attack    ~ Uniform[1, 10]  (integer)
  - arc cost_interdict ~ Uniform[1, 5]   (integer)
  - goal node reward   ~ Uniform[10, 50] (integer)

Parameter ranges used in the paper:
  L in {3, 4, 5}
  W in {3, 4, 5}
  d in {2, 3}
"""

from __future__ import annotations

import random
from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt

# add project root folder
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.attack_graph import AttackGraph


def generate_attack_graph(
    L: int,
    W: int,
    d: int,
    seed: Optional[int] = None,
) -> AttackGraph:
    """
    Generate a synthetic attack graph with L levels, W nodes per level,
    and out-degree d per node (connecting to the next level).

    Parameters
    ----------
    L    : int
        Number of levels EXCLUDING the source node at level 0.
        Total levels in the graph = L + 1  (0 to L).
        Goal nodes sit at level L.
    W    : int
        Number of nodes per level (levels 1 to L).
        Level 0 always has exactly 1 source node.
    d    : int
        Out-degree of each non-goal node.
        Each node at level l picks d distinct targets at level l+1.
        If d > W, it is clamped to W (connects to all nodes in next level).
    seed : int, optional
        Random seed for reproducibility. Pass the same seed to get the
        same graph; vary the seed to get the N=10 instances per parameter
        combination used in the paper's experiments.

    Returns
    -------
    AttackGraph
        A fully constructed AttackGraph ready to be passed to the solver.

    Notes
    -----
    Node id assignment:
        level 0 : node id = 0  (single source)
        level l : node ids = [ 1 + (l-1)*W + i  for i in 0..W-1 ]

    Examples
    --------
    g = generate_attack_graph(L=3, W=3, d=2, seed=42)
    print(g.summary())
    g = generate_attack_graph(L=4, W=4, d=3, seed=0)
    """

    rng = random.Random(seed)
    graph = AttackGraph()

    # Create nodes
    # Level 0: single source node (id=0), no reward
    graph.add_node(node_id=0, level=0, reward=0.0)

    # Levels 1 to L-1: intermediate nodes, no reward
    for level in range(1, L):
        for i in range(W):
            node_id = _node_id(level, i, W)
            graph.add_node(node_id=node_id, level=level, reward=0.0)

    # Level L: goal nodes, reward sampled from Uniform[10, 50]
    for i in range(W):
        node_id = _node_id(L, i, W)
        reward = float(rng.randint(10, 50))
        graph.add_node(node_id=node_id, level=L, reward=reward)

    # 2. Create arcs
    # Clamp d to W so we never ask for more targets than nodes available
    effective_d = min(d, W)

    # Level 0 -> Level 1
    next_level_ids = [_node_id(1, i, W) for i in range(W)]
    targets = rng.sample(next_level_ids, effective_d)
    for dst in targets:
        ca = float(rng.randint(1, 10))
        ci = float(rng.randint(1, 5))
        graph.add_arc(src=0, dst=dst, cost_attack=ca, cost_interdict=ci)

    # Level l -> Level l+1  for l = 1 to L-1
    for level in range(1, L):
        next_level_ids = [_node_id(level + 1, i, W) for i in range(W)]
        for i in range(W):
            src = _node_id(level, i, W)
            targets = rng.sample(next_level_ids, effective_d)
            for dst in targets:
                # Avoid duplicate arcs (can happen when d is close to W)
                if (src, dst) not in graph.arcs:
                    ca = float(rng.randint(1, 10))
                    ci = float(rng.randint(1, 5))
                    graph.add_arc(src=src, dst=dst, cost_attack=ca, cost_interdict=ci)

    return graph


# Internal helpers

def _node_id(level: int, index: int, W: int) -> int:
    """
    Compute the node id for the i-th node at the given level.

    Layout:
        level 0 -> id 0          (source)
        level 1 -> ids 1 to W
        level 2 -> ids W+1 to 2W
        level l -> ids 1+(l-1)*W to l*W
    """
    return 1 + (level - 1) * W + index


# Visualization

def draw_attack_graph(
    graph: AttackGraph,
    title: str = "Attack Graph",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Draw the attack graph with nodes arranged by level (left to right).

    Source nodes are green, goal nodes are red, intermediate nodes are
    light blue. Arc labels show attack cost and interdict cost.

    Parameters
    ----------
    graph     : AttackGraph  the graph to draw
    title     : str          plot title
    show      : bool         call plt.show() at the end
    save_path : str, optional  if given, save the figure to this path
    """
    g = graph._graph

    # Group nodes by level to compute positions
    levels: dict[int, list[int]] = {}
    for node_id, node in graph.nodes.items():
        levels.setdefault(node.level, []).append(node_id)

    # Position: x = level, y = rank within level (centred vertically)
    pos: dict[int, tuple[float, float]] = {}
    for level, node_ids in sorted(levels.items()):
        n = len(node_ids)
        for rank, node_id in enumerate(sorted(node_ids)):
            x = float(level)
            y = float(rank) - (n - 1) / 2.0
            pos[node_id] = (x, y)

    # Node colours
    node_colors = []
    goal_ids = {n.id for n in graph.goal_nodes}
    for node_id in g.nodes():
        node = graph.nodes[node_id]
        if node.level == 0:
            node_colors.append("#90EE90")   # green  - source
        elif node_id in goal_ids:
            node_colors.append("#FF6B6B")   # red    - goal
        else:
            node_colors.append("#AED6F1")   # blue   - intermediate

    # Node labels: show id and reward for goal nodes
    node_labels = {}
    for node_id, node in graph.nodes.items():
        if node.reward > 0:
            node_labels[node_id] = f"{node_id}\nr={node.reward:.0f}"
        else:
            node_labels[node_id] = str(node_id)

    # Edge labels: attack cost / interdict cost
    edge_labels = {
        (arc.src, arc.dst): f"a={arc.cost_attack:.0f}/i={arc.cost_interdict:.0f}"
        for arc in graph.arcs.values()
    }

    fig, ax = plt.subplots(figsize=(max(8, graph.num_levels * 3), 6))
    ax.set_title(title, fontsize=13, fontweight="bold")

    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=900, ax=ax)
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=8, ax=ax)
    nx.draw_networkx_edges(
        g, pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        edge_color="#444444",
        ax=ax,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_edge_labels(
        g, pos,
        edge_labels=edge_labels,
        font_size=7,
        ax=ax,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#90EE90", label="Source (level 0)"),
        Patch(facecolor="#AED6F1", label="Intermediate"),
        Patch(facecolor="#FF6B6B", label="Goal (level L)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# Batch generator - used in run_experiments.py

def generate_instances(
    L: int,
    W: int,
    d: int,
    n_instances: int = 10,
    base_seed: int = 0,
) -> list[AttackGraph]:
    """
    Generate n_instances independent graphs with the same (L, W, d)
    parameters but different random seeds.

    This matches the paper's experimental protocol: 10 instances per
    parameter combination, results averaged across instances.

    Parameters
    ----------
    L, W, d      : graph parameters (see generate_attack_graph)
    n_instances  : how many independent instances to generate (default 10)
    base_seed    : seeds used are base_seed, base_seed+1, ... base_seed+n-1

    Returns
    -------
    list[AttackGraph]
    """
    return [
        generate_attack_graph(L=L, W=W, d=d, seed=base_seed + i)
        for i in range(n_instances)
    ]


# Quick smoke test - run this file directly to verify the generator works

if __name__ == "__main__":
    print("Generating a small graph (L=3, W=3, d=2, seed=42)...")
    g = generate_attack_graph(L=3, W=3, d=2, seed=42)
    print(g.summary())
    print(f"  Arcs  : {list(g.arcs.keys())}")
    print(f"  Paths : {g.get_all_paths()}")

    print("\nGenerating 3 instances for (L=3, W=3, d=2)...")
    instances = generate_instances(L=3, W=3, d=2, n_instances=3)
    for i, inst in enumerate(instances):
        print(f"  Instance {i}: {inst}")

    print("\nDrawing graph (close window to exit)...")
    draw_attack_graph(g, title="Attack Graph - L=3, W=3, d=2, seed=42")