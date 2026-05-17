"""
model/bilevel.py

Small-instance exact bilevel formulation (MINMAXBREACH with path enumeration)
==============================================================================
For graphs small enough that all paths can be enumerated explicitly, this
module builds and solves the full bilevel model as a single MIP.

The trick: enumerate all attacker paths upfront, then write the min-max
as a linear program by introducing the variable L (worst-case breach loss)
and adding one constraint per path saying:
    L >= breach_reward(path, x)  if path is feasible for the attacker

This is valid because the inner attacker problem, given x, is a maximisation
over a finite set of paths - so the max equals the tightest lower bound on L.

Use this ONLY for validation on small instances (< ~20 nodes).
For larger instances use the CCG algorithm in model/algorithm.py.

Formulation (Section 3.1 - MINMAXBREACH, linearised via path enumeration):

    min   L

    s.t.  sum_{(i,j)} cost_interdict[i,j] * x[i,j] <= B_defender

          L >= reward[g(p)] * feasible(p, x)   for each path p
              where feasible(p, x) = 1 iff:
                (a) no arc of p is interdicted by x, AND
                (b) cost_attack(p) <= B_attacker

          x[i,j] in {0,1}
          L >= 0
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pulp

from model.attack_graph import AttackGraph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_bilevel_small(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], int], int]:
    """
    Solve the bilevel interdiction problem exactly by enumerating all paths.

    Only suitable for small graphs where the number of source-to-goal
    paths is manageable (typically fewer than a few thousand).

    Parameters
    ----------
    graph       : AttackGraph
    B_defender  : float  Defender's interdiction budget.
    B_attacker  : float  Attacker's traversal budget.
    solver_msg  : bool   Print CBC solver output if True.

    Returns
    -------
    breach_loss : float
        Optimal worst-case breach loss after interdiction.
    x_optimal   : dict  {(src, dst): 0 or 1}
        Optimal interdiction plan.
    n_paths     : int
        Number of feasible attacker paths enumerated.
    """
    # Enumerate all paths the attacker could take
    all_paths = graph.get_all_paths()

    # Keep only paths the attacker can afford (budget-feasible)
    feasible_paths = [
        p for p in all_paths
        if graph.path_cost_attack(p) <= B_attacker and graph.path_goal(p) is not None
    ]

    if not feasible_paths:
        # Attacker has no affordable path - breach loss is 0 regardless of x
        x_zero = {arc: 0 for arc in graph.arcs}
        return 0.0, x_zero, 0

    prob = pulp.LpProblem("MINMAXBREACH_small", pulp.LpMinimize)

    arcs = list(graph.arcs.keys())

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    # x[i,j] = 1 if the defender interdicts arc (i,j)
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for (i, j) in arcs
    }

    # L = worst-case breach loss (the variable we minimise)
    L = pulp.LpVariable("L", lowBound=0.0, cat="Continuous")

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    prob += L, "MinWorstCaseLoss"

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # 1. Defender budget
    prob += (
        pulp.lpSum(
            graph.arcs[(i, j)].cost_interdict * x[(i, j)]
            for (i, j) in arcs
        )
        <= B_defender,
        "DefenderBudget",
    )

    # 2. For each feasible attacker path: L >= reward * (1 if path not blocked)
    #
    # A path p is blocked iff at least one of its arcs is interdicted.
    # We model this with the big-M linearisation:
    #
    #   L >= reward[g] - M * sum_{(i,j) in p} x[i,j]
    #
    # where M = reward[g] is the maximum possible contribution of path p.
    # If any arc is interdicted, the RHS drops below 0 (non-binding).
    # If no arc is interdicted, the RHS = reward[g], forcing L >= reward[g].
    for idx, path in enumerate(feasible_paths):
        goal_id = graph.path_goal(path)
        reward  = graph.nodes[goal_id].reward
        path_arc_list = graph.path_arcs(path)

        prob += (
            L >= reward - reward * pulp.lpSum(x[(i, j)] for (i, j) in path_arc_list),
            f"PathConstraint_{idx}",
        )

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=1 if solver_msg else 0)
    prob.solve(solver)

    breach_loss = pulp.value(L) or 0.0
    x_optimal   = {(i, j): int(round(pulp.value(x[(i, j)]) or 0)) for (i, j) in arcs}

    return breach_loss, x_optimal, len(feasible_paths)