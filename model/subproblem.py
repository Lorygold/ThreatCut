"""
model/subproblem.py

Inner (attacker) problem: MAXBREACHBM
======================================
Given the defender's interdiction plan x (a dict mapping arc -> 0/1),
find the attack strategy that maximises total breach reward subject to
the attacker's budget B_attacker.

This is the "subproblem" in the constraint-and-column generation loop
(Algorithm 1 of the paper).  It is also used standalone to evaluate
how much damage an optimal attacker can do against any fixed plan x.

Formulation (Section 3.2 — MAXBREACHBM):

    max   sum_{g in Goals} reward[g] * z[g]

    s.t.  sum_{(i,j) in A} cost_attack[i,j] * (1 - x[i,j]) * y[i,j]
              <= B_attacker                            (attacker budget)

          sum_{(s,j)} y[s,j] <= 1   for each source s  (single-entry)

          sum_{(i,j)} y[i,j] - sum_{(j,k)} y[j,k] >= 0
              for each intermediate node j             (flow continuity)

          z[g] <= sum_{(i,g)} y[i,g]   for each goal g  (goal reachability)

          y[i,j] <= 1 - x[i,j]   for all (i,j)          (interdicted arcs)

          y[i,j] in {0,1},  z[g] in {0,1}
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import pulp

from model.attack_graph import AttackGraph


# Public API

def solve_attacker_problem(
    graph: AttackGraph,
    x_interdict: Dict[Tuple[int, int], int],
    B_attacker: float,
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], float], List[List[int]]]:
    """
    Solve the attacker's inner optimisation problem (MAXBREACHBM).

    Parameters
    ----------
    graph        : AttackGraph
    x_interdict  : dict  {(src, dst): 0 or 1}
                   Defender's interdiction plan.  Missing keys treated as 0.
    B_attacker   : float  Attacker's budget.
    solver_msg   : bool   If True, print CBC solver output.

    Returns
    -------
    obj_value    : float  Optimal breach reward (0 if attacker cannot breach).
    y_solution   : dict   {(src, dst): float}  arc flow values in [0,1].
    attack_paths : list   List of paths used by the optimal attacker.
                          Each path is a list of node ids.
    """
    prob = pulp.LpProblem("MAXBREACHBM", pulp.LpMaximize)

    arcs = list(graph.arcs.keys())
    goals = graph.goal_nodes
    sources = graph.source_nodes

    # Decision variables
    # y[i,j] = 1 if attacker uses arc (i,j)
    y = {
        (i, j): pulp.LpVariable(f"y_{i}_{j}", cat="Binary")
        for (i, j) in arcs
    }

    # z[g] = 1 if goal node g is breached
    z = {
        g.id: pulp.LpVariable(f"z_{g.id}", cat="Binary")
        for g in goals
    }

    # Objective: maximise total breach reward
    prob += pulp.lpSum(g.reward * z[g.id] for g in goals), "MaxBreach"

    # Constraints

    # 1. Attacker budget: only pay for arcs that are NOT interdicted
    prob += (
        pulp.lpSum(
            graph.arcs[(i, j)].cost_attack
            * (1 - x_interdict.get((i, j), 0))
            * y[(i, j)]
            for (i, j) in arcs
        )
        <= B_attacker,
        "AttackerBudget",
    )

    # 2. Interdicted arcs cannot be used
    for (i, j) in arcs:
        if x_interdict.get((i, j), 0) == 1:
            prob += y[(i, j)] == 0, f"Interdicted_{i}_{j}"

    # 3. Single entry from source nodes
    for src in sources:
        out_arcs = [(i, j) for (i, j) in arcs if i == src.id]
        if out_arcs:
            prob += (
                pulp.lpSum(y[(i, j)] for (i, j) in out_arcs) <= 1,
                f"SingleEntry_{src.id}",
            )

    # 4. Flow continuity at intermediate nodes
    intermediate_ids = {
        n.id for n in graph.nodes.values()
        if n.level > 0 and n.id not in {g.id for g in goals}
    }
    for node_id in intermediate_ids:
        in_arcs  = [(i, j) for (i, j) in arcs if j == node_id]
        out_arcs = [(i, j) for (i, j) in arcs if i == node_id]
        if in_arcs and out_arcs:
            prob += (
                pulp.lpSum(y[(i, j)] for (i, j) in in_arcs)
                >= pulp.lpSum(y[(i, j)] for (i, j) in out_arcs),
                f"FlowCont_{node_id}",
            )

    # 5. Goal reachability: z[g]=1 only if some arc enters g
    for g in goals:
        in_arcs = [(i, j) for (i, j) in arcs if j == g.id]
        if in_arcs:
            prob += (
                z[g.id] <= pulp.lpSum(y[(i, j)] for (i, j) in in_arcs),
                f"GoalReach_{g.id}",
            )
        else:
            prob += z[g.id] == 0, f"GoalUnreachable_{g.id}"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=1 if solver_msg else 0)
    prob.solve(solver)

    obj_value = pulp.value(prob.objective) or 0.0

    y_solution = {
        (i, j): pulp.value(y[(i, j)]) or 0.0
        for (i, j) in arcs
    }

    attack_paths = _extract_paths(graph, y_solution)

    return obj_value, y_solution, attack_paths


# Helper: reconstruct paths from arc flow values

def _extract_paths(
    graph: AttackGraph,
    y_solution: Dict[Tuple[int, int], float],
    threshold: float = 0.5,
) -> List[List[int]]:
    """
    Reconstruct the set of attack paths from the arc flow solution.

    An arc is considered "used" if its y value exceeds threshold (0.5).
    Starting from each source node, we do a DFS following used arcs
    until we reach a goal node or a dead end.

    Parameters
    ----------
    graph      : AttackGraph
    y_solution : dict  arc -> float value from the solver
    threshold  : float  treat arc as used if value >= threshold

    Returns
    -------
    List of paths (each path is a list of node ids).
    """
    used_arcs = {(i, j) for (i, j), v in y_solution.items() if v >= threshold}
    goal_ids  = {g.id for g in graph.goal_nodes}
    paths: List[List[int]] = []

    def dfs(node: int, current_path: List[int]) -> None:
        out = [(i, j) for (i, j) in used_arcs if i == node]
        if not out:
            if node in goal_ids:
                paths.append(list(current_path))
            return
        for (i, j) in out:
            current_path.append(j)
            dfs(j, current_path)
            current_path.pop()

    for src in graph.source_nodes:
        dfs(src.id, [src.id])

    return paths