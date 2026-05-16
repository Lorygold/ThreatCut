"""
model/attack_graph.py

Core data structures for the attack graph.
No solver dependency – pure Python dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Node:
    """A node in the attack graph (vulnerability, transition, or goal)."""
    node_id: int
    reward: float = 0.0   # breach loss / attacker reward; > 0 only for goal nodes


@dataclass
class Arc:
    """A directed arc representing an attacker action or exploit."""
    tail: int
    head: int
    cost_attack: float = 1.0       # cost for the attacker to traverse this arc
    cost_interdict: float = 1.0    # cost for the defender to interdict this arc


@dataclass
class AttackGraph:
    """
    Container for the full attack graph G = (N, A).

    nodes : dict  node_id       -> Node
    arcs  : dict  (tail, head)  -> Arc
    """
    nodes: Dict[int, Node] = field(default_factory=dict)
    arcs:  Dict[Tuple[int, int], Arc] = field(default_factory=dict)

    def add_node(self, node_id: int, reward: float = 0.0) -> None:
        self.nodes[node_id] = Node(node_id=node_id, reward=reward)

    def add_arc(
        self,
        tail: int,
        head: int,
        cost_attack: float = 1.0,
        cost_interdict: float = 1.0,
    ) -> None:
        self.arcs[(tail, head)] = Arc(
            tail=tail,
            head=head,
            cost_attack=cost_attack,
            cost_interdict=cost_interdict,
        )
