"""
data/generator.py

Synthetic attack graph generator following the experimental setup of:
    Nandi, Medal, Vadlamani (2016)
    "Interdicting Attack Graphs to Protect Organizations from Cyber Attacks"
    Computers & Operations Research, Vol. 75, pp. 118-131

Graph structure (Section 5 - Computational Experiments, Fig. 4):
  - 1 source node at level 0 (models the set of vulnerability entry points)
  - W nodes per level, for L levels (levels 1 … L)
  - d  non-source arcs drawn uniformly at random between consecutive levels
       (actual count may be slightly lower when the graph is nearly saturated)
  - arc cost_attack    ~ Uniform[mina, maxa]
  - arc cost_interdict ~ Uniform[mind, maxd]
  - goal node reward   ~ Uniform[minr, maxr]

The source node connects to ALL W nodes at level 1 with cost_attack = 0 and
an effectively infinite cost_interdict, modelling initial system vulnerabilities
that the defender cannot block.

Paper parameter ranges (Table 2):
  - Nodes    : 50, 100, 150, 200        (≈ 1 + L × W)
  - Arcs     : ≈ 2.15 × total_nodes     (set d ≈ 2.15 × nodes − W)
  - Levels   : 5 or 7
  - Rewards  : Uniform(500, 1500) "low"  |  Uniform(1000, 2000) "high"
  - Attack   : Uniform(10, 30)   "low"  |  Uniform(30, 50)     "high"
  - Interdict: Uniform(10, 30)   "low"  |  Uniform(30, 50)     "high"

Convenience presets matching the paper:
  LOW_COSTS  = dict(mina=10, maxa=30, mind=10, maxd=30, minr=500,  maxr=1500)
  HIGH_COSTS = dict(mina=30, maxa=50, mind=30, maxd=50, minr=1000, maxr=2000)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.attack_graph import AttackGraph


# Cost/reward presets matching the paper's "low" and "high" parameter levels

LOW_COSTS  = dict(mina=10, maxa=30, mind=10, maxd=30, minr=500,  maxr=1500)
HIGH_COSTS = dict(mina=30, maxa=50, mind=30, maxd=50, minr=1000, maxr=2000)

# Source arcs are structurally unblockable (initial system vulnerabilities).
_SOURCE_CI = 99_999


def generate_attack_graph(
    L: int,
    W: int,
    d: int,
    mina: int,
    maxa: int,
    mind: int,
    maxd: int,
    minr: int,
    maxr: int,
    seed: Optional[int] = None,
) -> AttackGraph:
    """
    Generate a synthetic attack graph with L levels and W nodes per level.

    Parameters
    ----------
    L    : int   Levels EXCLUDING the source at level 0 (goal nodes at level L).
    W    : int   Nodes per level (levels 1 … L).
    d    : int   Target number of non-source arcs.  Arcs are drawn uniformly
                 at random between consecutive levels.  The actual arc count
                 equals d unless the inter-level graph is already saturated.
                 For the paper's density (≈ 2.15 × nodes): d ≈ 2.15*(1+L*W) − W.
    mina, maxa   : int  Attack-cost range  ~ Uniform[mina, maxa].
    mind, maxd   : int  Interdict-cost range ~ Uniform[mind, maxd].
    minr, maxr   : int  Goal-reward range  ~ Uniform[minr, maxr].
    seed : int, optional   Random seed for reproducibility.

    Returns
    -------
    AttackGraph

    Notes
    -----
    Node id layout:
        level 0 → id 0            (single source)
        level l → ids 1+(l-1)*W … l*W
    """
    rng = random.Random(seed)
    graph = AttackGraph()

    # --- 1. Nodes ---
    graph.add_node(0, level=0, reward=0.0)

    for level in range(1, L):
        for i in range(W):
            graph.add_node(_node_id(level, i, W), level=level, reward=0.0)

    for i in range(W):
        nid = _node_id(L, i, W)
        graph.add_node(nid, level=L, reward=float(rng.randint(minr, maxr)))

    # --- 2. Source → all level-1 nodes (ca=0, ci=large) ---
    for i in range(W):
        graph.add_arc(0, _node_id(1, i, W),
                      cost_attack=0.0, cost_interdict=float(_SOURCE_CI))

    # --- 3. Guarantee connectivity: every level-(l+1) node gets ≥1 incoming arc ---
    # Randomly match each level-(l+1) node to a distinct level-l node (round-robin
    # with a random offset so the matching itself is random).
    for level in range(1, L):
        src_ids = [_node_id(level,   i, W) for i in range(W)]
        dst_ids = [_node_id(level+1, i, W) for i in range(W)]
        offset  = rng.randrange(W)
        for k, dst in enumerate(dst_ids):
            src = src_ids[(k + offset) % W]
            if (src, dst) not in graph.arcs:
                ca = float(rng.randint(mina, maxa))
                ci = float(rng.randint(mind, maxd))
                graph.add_arc(src, dst, ca, ci)

    # --- 4. Add random inter-level arcs until non-source arc count reaches d ---
    def _non_source_count() -> int:
        return len(graph.arcs) - W

    if _non_source_count() < d:
        # Build candidate pool: all inter-level pairs not yet present.
        candidates = [
            (src, dst)
            for level in range(1, L)
            for src in (_node_id(level,   i, W) for i in range(W))
            for dst in (_node_id(level+1, j, W) for j in range(W))
            if (src, dst) not in graph.arcs
        ]
        # RM in - this includes arcs between nodes of the last level
        for src in (_node_id(L, i, W) for i in range(W)):
            for dst in (_node_id(L, j, W) for j in range(W)):
                if src!=dst: candidates.append((src,dst))
        # RM out
        
        rng.shuffle(candidates)
        for src, dst in candidates:
            if _non_source_count() >= d:
                break
            if (src, dst) not in graph.arcs:        # re-check after earlier insertions
                ca = float(rng.randint(mina, maxa))
                ci = float(rng.randint(mind, maxd))
                graph.add_arc(src, dst, ca, ci)

    return graph


# Internal helpers

def _node_id(level: int, index: int, W: int) -> int:
    """
    Node id for the i-th node at the given level.

    Layout:
        level 0 → id 0
        level l → ids 1+(l-1)*W … l*W
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
    """
    g = graph._graph

    levels: dict[int, list[int]] = {}
    for node_id, node in graph.nodes.items():
        levels.setdefault(node.level, []).append(node_id)

    pos: dict[int, tuple[float, float]] = {}
    for level, node_ids in sorted(levels.items()):
        n = len(node_ids)
        for rank, node_id in enumerate(sorted(node_ids)):
            pos[node_id] = (float(level), float(rank) - (n - 1) / 2.0)

    goal_ids = {n.id for n in graph.goal_nodes}
    node_colors = []
    for node_id in g.nodes():
        node = graph.nodes[node_id]
        if node.level == 0:
            node_colors.append("#90EE90")
        elif node_id in goal_ids:
            node_colors.append("#FF6B6B")
        else:
            node_colors.append("#AED6F1")

    node_labels = {
        node_id: (f"{node_id}\nr={node.reward:.0f}" if node.reward > 0 else str(node_id))
        for node_id, node in graph.nodes.items()
    }

    edge_labels = {
        (arc.src, arc.dst): f"a={arc.cost_attack:.0f}/i={arc.cost_interdict:.0f}"
        for arc in graph.arcs.values()
        if arc.cost_interdict < _SOURCE_CI
    }

    fig, ax = plt.subplots(figsize=(max(8, graph.num_levels * 3), 6))
    ax.set_title(title, fontsize=13, fontweight="bold")
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=900, ax=ax)
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=8, ax=ax)
    nx.draw_networkx_edges(g, pos, arrows=True, arrowstyle="-|>", arrowsize=20,
                           edge_color="#444444", ax=ax, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#90EE90", label="Source (level 0)"),
        Patch(facecolor="#AED6F1", label="Intermediate"),
        Patch(facecolor="#FF6B6B", label="Goal (level L)"),
    ], loc="upper left", fontsize=9)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# Batch generator

def generate_instances(
    L: int,
    W: int,
    d: int,
    mina: int,
    maxa: int,
    mind: int,
    maxd: int,
    minr: int,
    maxr: int,
    n_instances: int = 10,
    base_seed: int = 0,
) -> list[AttackGraph]:
    """
    Generate n_instances independent graphs with the same parameters but
    different random seeds (matches the paper's protocol of 10 instances
    per parameter combination).
    """
    return [
        generate_attack_graph(
            L=L, W=W, d=d,
            mina=mina, maxa=maxa,
            mind=mind, maxd=maxd,
            minr=minr, maxr=maxr,
            seed=base_seed + i,
        )
        for i in range(n_instances)
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate and display a synthetic attack graph.\n\n"
            "Paper-scale examples (Table 2, low cost preset):\n"
            "  ~50  nodes, 5 levels:  --L 5 --W 10 --d 100 --B_def 75  --B_att 125\n"
            "  ~50  nodes, 7 levels:  --L 7 --W  7 --d 100 --B_def 75  --B_att 125\n"
            "  ~100 nodes, 5 levels:  --L 5 --W 20 --d 197 --B_def 150 --B_att 150\n"
            "  ~150 nodes, 5 levels:  --L 5 --W 30 --d 295 --B_def 275 --B_att 325\n"
            "  ~200 nodes, 5 levels:  --L 5 --W 40 --d 393 --B_def 375 --B_att 425\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--L",    type=int, default=3,   help="Number of levels (default 3)")
    parser.add_argument("--W",    type=int, default=3,   help="Nodes per level (default 3)")
    parser.add_argument("--d",    type=int, default=18,  help="Target non-source arc count (default 18)")
    parser.add_argument("--seed", type=int, default=0,   help="Random seed (default 0)")
    parser.add_argument("--high_costs", action="store_true",
                        help="Use paper's 'high' cost preset instead of 'low'")
    parser.add_argument("--no_draw", action="store_true",
                        help="Skip the graph visualisation (useful for large graphs)")
    args = parser.parse_args()

    costs = HIGH_COSTS if args.high_costs else LOW_COSTS
    preset_name = "high" if args.high_costs else "low"

    total_nodes = 1 + args.L * args.W
    print(f"Generating graph: L={args.L}, W={args.W}, d={args.d}, "
          f"nodes≈{total_nodes}, seed={args.seed}, costs={preset_name}")
    g = generate_attack_graph(L=args.L, W=args.W, d=args.d, seed=args.seed, **costs)
    print(g.summary())
    print(f"  Arcs  : {len(g.arcs)} total  "
          f"(source: {args.W}, inter-level: {len(g.arcs) - args.W})")
    if args.L <= 4 and args.W <= 6:
        print(f"  Paths : {g.get_all_paths()}")

    if not args.no_draw:
        title = f"Attack Graph - L={args.L}, W={args.W}, d={args.d} ({preset_name} costs)"
        draw_attack_graph(g, title=title)
