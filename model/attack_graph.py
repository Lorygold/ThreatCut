"""
model/attack_graph.py

Data structures for the attack graph used by all three models.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Node:
    """Represents a node in the attack graph."""
    node_id: int
    reward: float = 0.0        # Loss / reward if this goal node is breached


@dataclass
class Arc:
    """Represents a directed arc (attacker action / exploit) in the attack graph."""
    tail: int                  # Source node
    head: int                  # Destination node
    cost_attack: float = 1.0   # Cost for the attacker to traverse this arc
    cost_interdict: float = 1.0  # Cost for the defender to interdict this arc


@dataclass
class AttackGraph:
    """
    Container for the full attack graph G = (N, A).

    Attributes
    ----------
    nodes : dict mapping node_id -> Node
    arcs  : dict mapping (tail, head) -> Arc
    """
    nodes: Dict[int, Node] = field(default_factory=dict)
    arcs: Dict[Tuple[int, int], Arc] = field(default_factory=dict)

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
