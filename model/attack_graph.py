"""
Data structure for the attack graph used in:
    Nandi, Medal, Vadlamani (2016)
    "Interdicting Attack Graphs to Protect Organizations from Cyber Attacks"
    Computers & Operations Research, Vol. 75, pp. 118-131

An attack graph is a directed acyclic graph (DAG) with a hierarchical
level structure:
  - Level 0:   source nodes (initially vulnerable entry points)
  - Level 1..L-1: intermediate nodes (privilege levels / access states)
  - Level L:   goal nodes (critical assets the attacker wants to breach)

Each arc (i, j) carries:
  - cost_attack    : cost the attacker pays to traverse the arc
  - cost_interdict : cost the defender pays to block (interdict) the arc

Each goal node carries:
  - reward : value gained by the attacker (= loss suffered by the defender)
             if that node is breached
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx


# Node and Arc descriptors

@dataclass
class Node:
    id: int
    level: int
    reward: float = 0.0  # non-zero only for goal nodes (last level)

@dataclass
class Arc:
    src: int
    dst: int
    cost_attack: float  # attacker's cost to traverse this arc
    cost_interdict: float  # defender's cost to block this arc


# AttackGraph

class AttackGraph:
    """
    Directed acyclic graph representing all possible attack paths
    through an information system.

    Internally backed by a networkx.DiGraph so that path enumeration
    and graph algorithms come for free.

    Attributes
    ----------
    _graph : nx.DiGraph
        The underlying directed graph. Node attributes store the Node
        dataclass; edge attributes store the Arc dataclass.
    _num_levels : int
        Total number of levels (set automatically when nodes are added).
    """

    def __init__(self) -> None:
        """
        Initialize an empty attack graph.

        Internal state
        --------------
        _graph       : empty DiGraph - nodes and arcs added via add_node / add_arc
        _nodes       : dict mapping node_id -> Node dataclass
        _arcs        : dict mapping (src, dst) -> Arc dataclass
        _num_levels  : tracks the maximum level seen so far; updated in add_node
        """
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[int, Node] = {}
        self._arcs: Dict[Tuple[int, int], Arc] = {}
        self._num_levels: int = 0   # Tracks the highest level index seen, used by goal_nodes


    # Graph construction

    def add_node(self, node_id: int, level: int, reward: float = 0.0) -> None:
        """
        Add a node to the attack graph.

        Parameters
        ----------
        node_id : int
            Unique integer identifier for the node.
        level   : int
            Hierarchy level.  0 = entry point, max level = goal.
        reward  : float
            Reward for the attacker if this node is breached.
            Should be > 0 only for goal nodes (last level).
        """
        node = Node(id=node_id, level=level, reward=reward)
        self._nodes[node_id] = node
        self._graph.add_node(node_id, data=node)

        if level > self._num_levels:
            self._num_levels = level

    def add_arc(
            self,
            src: int,
            dst: int,
            cost_attack: float,
            cost_interdict: float,
    ) -> None:
        """
        Add a directed arc from src to dst.

        Parameters
        ----------
        src             : int   source node id
        dst             : int   destination node id
        cost_attack     : float attacker's cost to use this arc
        cost_interdict  : float defender's cost to block this arc
        """
        if src not in self._nodes:
            raise ValueError(f"Source node_id {src} not in graph. Add it first.")
        if dst not in self._nodes:
            raise ValueError(f"Destination node_id {dst} not in graph. Add it first.")

        arc = Arc(src=src, dst=dst, cost_attack=cost_attack, cost_interdict=cost_interdict)
        self._arcs[(src, dst)] = arc
        self._graph.add_edge(src, dst, data=arc)


    # Properties

    @property
    def nodes(self) -> Dict[int, Node]:
        """All nodes keyed by node_id."""
        return self._nodes

    @property
    def arcs(self) -> Dict[Tuple[int, int], Arc]:
        """All arcs keyed by (src, dst) tuple."""
        return self._arcs

    @property
    def source_nodes(self) -> List[Node]:
        """Nodes at level 0 - the attacker's entry points."""
        return [n for n in self._nodes.values() if n.level == 0]

    @property
    def goal_nodes(self) -> List[Node]:
        """Nodes at the highest level - the critical assets to protect."""
        return [n for n in self._nodes.values() if n.level == self._num_levels]

    @property
    def num_levels(self) -> int:
        """Total number of distinct levels in the graph."""
        return self._num_levels + 1  # levels are 0-indexed


    # Paths

    def get_paths_to_goal(self, goal_node_id: int) -> List[List[int]]:
        """
        Return all simple paths from any source node to goal_node_id.

        Each path is a list of node ids, e.g. [0, 2, 5, 7].
        These paths become the *columns* in the master problem of the
        constraint-and-column generation algorithm.

        Parameters
        ----------
        goal_node_id : int
            The id of a goal node.

        Returns
        -------
        List[List[int]]
            All simple paths leading to goal_node_id, one list per path.
        """
        paths: List[List[int]] = []
        for src in self.source_nodes:
            for path in nx.all_simple_paths(self._graph, src.id, goal_node_id):
                paths.append(path)
        return paths

    def get_all_paths(self) -> List[List[int]]:
        """
        Return all simple paths from any source node to any goal node.

        Useful for the small-instance bilevel formulation (Step 4).
        """
        paths: List[List[int]] = []
        for goal in self.goal_nodes:
            paths.extend(self.get_paths_to_goal(goal.id))
        return paths

    def path_arcs(self, path: List[int]) -> List[Tuple[int, int]]:
        """
        Convert a path (list of node ids) to the list of arcs it traverses.

        Example: [0, 2, 5] -> [(0,2), (2,5)]
        """
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def path_cost_attack(self, path: List[int]) -> float:
        """Total attack cost for a given path."""
        return sum(self._arcs[(u, v)].cost_attack for u, v in self.path_arcs(path))

    def path_goal(self, path: List[int]) -> Optional[int]:
        """
        Return the goal node id at the end of the path,
        or None if the last node is not a goal node.
        """
        last = path[-1]
        if last in self._nodes and self._nodes[last].level == self._num_levels:
            return last
        return None

    def is_path_blocked(self, path: List[int], x: Dict[Tuple[int, int], int]) -> bool:
        """
        Return True if at least one arc in the path is interdicted by x.

        Parameters
        ----------
        path : List[int]    sequence of node ids
        x    : dict         interdiction plan - x[(i,j)] = 1 if arc is blocked
        """
        return any(x.get((u, v), 0) == 1 for u, v in self.path_arcs(path))


    # Display

    def summary(self) -> str:
        """Human-readable summary of the graph."""
        lines = [
            f"AttackGraph: {len(self._nodes)} nodes, {len(self._arcs)} arcs, "
            f"{self.num_levels} levels",
            f"  Source nodes : {[n.id for n in self.source_nodes]}",
            f"  Goal nodes   : {[n.id for n in self.goal_nodes]} "
            f"(rewards: {[n.reward for n in self.goal_nodes]})",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"AttackGraph(nodes={len(self._nodes)}, "
            f"arcs={len(self._arcs)}, "
            f"levels={self.num_levels})"
        )
