"""
model/algorithm.py

Constraint-and-Column Generation (CCG) algorithm + two heuristics
==================================================================
Implementation of Algorithm 1 from:
    Nandi, Medal, Vadlamani (2016)
    "Interdicting Attack Graphs to Protect Organizations from Cyber Attacks"
    Computers & Operations Research, Vol. 75, pp. 118-131

The CCG algorithm iteratively:
  1. Solves the MASTER problem (defender) with the paths known so far
     → obtains a lower bound and an interdiction plan x_hat
  2. Solves the SUBPROBLEM (attacker) given x_hat
     → obtains an upper bound (worst-case breach against x_hat)
  3. If UB - LB <= epsilon: stop, x_hat is optimal
  4. Otherwise: add the worst-case attack path as a new column/constraint
     to the master, then go to step 1.

Two heuristics are also implemented (Section 4 of the paper):
  - run_heuristic_lp:     relaxes the master to an LP at each iteration
  - run_heuristic_greedy: replaces the master with a greedy budget allocation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pulp

from model.attack_graph import AttackGraph
from model.subproblem import solve_attacker_problem


# Result dataclass

@dataclass
class CCGResult:
    """Holds all output produced by the CCG algorithm or a heuristic."""
    # Optimal / best-found interdiction plan  {(i,j): 0 or 1}
    x_optimal: Dict[Tuple[int, int], int]

    # Worst-case breach loss under x_optimal (= upper bound at convergence)
    breach_loss: float

    # Number of CCG iterations performed
    n_iterations: int

    # Lower bound after each iteration (master objective)
    lower_bounds: List[float] = field(default_factory=list)

    # Upper bound after each iteration (subproblem objective)
    upper_bounds: List[float] = field(default_factory=list)

    # Wall-clock time in seconds
    solve_time_s: float = 0.0

    # Paths added to the master during the algorithm
    paths_added: List[List[int]] = field(default_factory=list)

    # Gap at termination: (UB - LB) / UB  (0 for exact algorithm)
    optimality_gap: float = 0.0


# CCG — exact algorithm

def run_ccg_algorithm(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    epsilon: float = 1e-4,
    max_iter: int = 200,
    solver_msg: bool = False,
) -> CCGResult:
    """
    Solve the bilevel interdiction problem exactly via CCG (Algorithm 1).

    Parameters
    ----------
    graph       : AttackGraph
    B_defender  : float   Defender's interdiction budget.
    B_attacker  : float   Attacker's traversal budget.
    epsilon     : float   Convergence tolerance (UB - LB <= epsilon → stop).
    max_iter    : int     Safety cap on the number of iterations.
    solver_msg  : bool    Print CBC output if True.

    Returns
    -------
    CCGResult
    """
    t0 = time.perf_counter()

    arcs = list(graph.arcs.keys())
    paths: List[List[int]] = []   # columns added so far

    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    x_hat: Dict[Tuple[int, int], int] = {a: 0 for a in arcs}
    ub = float("inf")
    lb = 0.0

    for iteration in range(1, max_iter + 1):

        # Step 1: solve master problem (defender)
        lb, x_hat = _solve_master(
            graph, paths, B_defender, B_attacker,
            integer=True, solver_msg=solver_msg
        )

        # Step 2: solve subproblem (attacker) given x_hat
        ub, _, attack_paths = solve_attacker_problem(
            graph, x_hat, B_attacker, solver_msg=solver_msg
        )

        lower_bounds.append(lb)
        upper_bounds.append(ub)

        # Step 3: convergence check
        if ub - lb <= epsilon:
            break

        # Step 4: add worst-case path(s) as new columns
        added = False
        for path in attack_paths:
            if path and path not in paths:
                paths.append(path)
                added = True

        # If subproblem found no new path (shouldn't happen but guard it)
        if not added:
            break

    gap = (ub - lb) / max(ub, 1e-9) if ub > 1e-9 else 0.0

    return CCGResult(
        x_optimal=x_hat,
        breach_loss=ub,
        n_iterations=iteration,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        solve_time_s=time.perf_counter() - t0,
        paths_added=list(paths),
        optimality_gap=gap,
    )


# Heuristic 1 — LP relaxation of master (one-node B&B)

def run_heuristic_lp(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    epsilon: float = 1e-4,
    max_iter: int = 200,
    solver_msg: bool = False,
) -> CCGResult:
    """
    CCG heuristic: relax the master problem to an LP at each iteration.

    The master's binary variables x[i,j] are relaxed to [0,1] continuous.
    The resulting x_hat may be fractional; we round greedily before passing
    it to the subproblem.  This is faster than the exact MIP master but may
    return a sub-optimal interdiction plan.

    Parameters and return value are identical to run_ccg_algorithm.
    """
    t0 = time.perf_counter()

    arcs   = list(graph.arcs.keys())
    paths:  List[List[int]] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    x_hat: Dict[Tuple[int, int], int] = {a: 0 for a in arcs}
    ub = float("inf")
    lb = 0.0

    for iteration in range(1, max_iter + 1):

        # LP relaxation of master
        lb, x_frac = _solve_master(
            graph, paths, B_defender, B_attacker,
            integer=False, solver_msg=solver_msg
        )

        # Round fractional solution greedily (keep arcs sorted by
        # fractional value descending, add until budget exhausted)
        x_hat = _greedy_round(graph, x_frac, B_defender)

        ub, _, attack_paths = solve_attacker_problem(
            graph, x_hat, B_attacker, solver_msg=solver_msg
        )

        lower_bounds.append(lb)
        upper_bounds.append(ub)

        if ub - lb <= epsilon:
            break

        added = False
        for path in attack_paths:
            if path and path not in paths:
                paths.append(path)
                added = True
        if not added:
            break

    gap = (ub - lb) / max(ub, 1e-9) if ub > 1e-9 else 0.0

    return CCGResult(
        x_optimal=x_hat,
        breach_loss=ub,
        n_iterations=iteration,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        solve_time_s=time.perf_counter() - t0,
        paths_added=list(paths),
        optimality_gap=gap,
    )


# Heuristic 2 — greedy master

def run_heuristic_greedy(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    epsilon: float = 1e-4,
    max_iter: int = 200,
    solver_msg: bool = False,
) -> CCGResult:
    """
    CCG heuristic: replace the master entirely with a greedy arc selection.

    At each iteration the defender greedily interdicts arcs in order of
    decreasing  (cost_interdict / cost_attack) ratio — i.e. arcs that are
    cheap to block but expensive for the attacker to traverse.

    Much faster than the exact master but typically gives a larger gap.
    """
    t0 = time.perf_counter()

    arcs   = list(graph.arcs.keys())
    paths:  List[List[int]] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    x_hat = _greedy_interdict(graph, B_defender)
    ub    = float("inf")
    lb    = 0.0

    for iteration in range(1, max_iter + 1):

        ub, _, attack_paths = solve_attacker_problem(
            graph, x_hat, B_attacker, solver_msg=solver_msg
        )

        # Lower bound: use the subproblem value as a proxy
        # (greedy does not produce a proper LB — record 0)
        lb = 0.0
        lower_bounds.append(lb)
        upper_bounds.append(ub)

        added = False
        for path in attack_paths:
            if path and path not in paths:
                paths.append(path)
                added = True

        # Re-run greedy with the same budget (plan does not change —
        # this heuristic converges in 1 iteration by design)
        if not added or iteration > 1:
            break

    gap = (ub - lb) / max(ub, 1e-9) if ub > 1e-9 else 0.0

    return CCGResult(
        x_optimal=x_hat,
        breach_loss=ub,
        n_iterations=iteration,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        solve_time_s=time.perf_counter() - t0,
        paths_added=list(paths),
        optimality_gap=gap,
    )


# Master problem builder

def _solve_master(
    graph: AttackGraph,
    paths: List[List[int]],
    B_defender: float,
    B_attacker: float,
    integer: bool = True,
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """
    Build and solve the master (defender) problem with the given set of paths.

    The master minimises L (worst-case breach loss) subject to:
      - Defender budget on interdiction costs
      - For each known attacker path p:
          L >= reward[g(p)] - reward[g(p)] * sum_{(i,j) in p} x[i,j]
        (big-M linearisation: path p is blocked if any arc is interdicted)

    Parameters
    ----------
    graph      : AttackGraph
    paths      : list of paths already added as columns
    B_defender : float
    B_attacker : float  (used to filter infeasible paths)
    integer    : bool   True = MIP master, False = LP relaxation
    solver_msg : bool

    Returns
    -------
    (obj_value, x_solution)
        obj_value   : float  master objective (= lower bound)
        x_solution  : dict   {arc: float or int} interdiction values
    """
    prob = pulp.LpProblem("MasterProblem", pulp.LpMinimize)
    arcs = list(graph.arcs.keys())

    cat = "Binary" if integer else "Continuous"

    x = {
        (i, j): pulp.LpVariable(
            f"x_{i}_{j}",
            lowBound=0.0,
            upBound=1.0,
            cat=cat,
        )
        for (i, j) in arcs
    }

    L = pulp.LpVariable("L", lowBound=0.0, cat="Continuous")

    # Objective
    prob += L, "MinWorstCase"

    # Defender budget
    prob += (
        pulp.lpSum(
            graph.arcs[(i, j)].cost_interdict * x[(i, j)]
            for (i, j) in arcs
        )
        <= B_defender,
        "DefenderBudget",
    )

    # One constraint per known attacker path
    for idx, path in enumerate(paths):
        goal_id = graph.path_goal(path)
        if goal_id is None:
            continue
        if graph.path_cost_attack(path) > B_attacker:
            continue   # path is not affordable — skip

        reward     = graph.nodes[goal_id].reward
        path_arcs  = graph.path_arcs(path)

        prob += (
            L >= reward - reward * pulp.lpSum(x[(i, j)] for (i, j) in path_arcs),
            f"PathCut_{idx}",
        )

    solver = pulp.PULP_CBC_CMD(msg=1 if solver_msg else 0)
    prob.solve(solver)

    obj = pulp.value(prob.objective) or 0.0
    x_sol = {(i, j): pulp.value(x[(i, j)]) or 0.0 for (i, j) in arcs}

    return obj, x_sol


# Greedy helpers

def _greedy_round(
    graph: AttackGraph,
    x_frac: Dict[Tuple[int, int], float],
    B_defender: float,
) -> Dict[Tuple[int, int], int]:
    """
    Round a fractional interdiction solution to integer by selecting arcs
    in decreasing order of fractional value until the budget is exhausted.
    """
    arcs = list(graph.arcs.keys())
    sorted_arcs = sorted(arcs, key=lambda a: x_frac.get(a, 0.0), reverse=True)

    x_int   = {a: 0 for a in arcs}
    budget  = B_defender

    for arc in sorted_arcs:
        ci = graph.arcs[arc].cost_interdict
        if ci <= budget:
            x_int[arc] = 1
            budget -= ci

    return x_int


def _greedy_interdict(
    graph: AttackGraph,
    B_defender: float,
) -> Dict[Tuple[int, int], int]:
    """
    Pure greedy: sort arcs by cost_interdict / cost_attack (ascending)
    — cheap-to-block, expensive-to-traverse arcs first.
    """
    arcs = list(graph.arcs.keys())

    def priority(arc: Tuple[int, int]) -> float:
        a = graph.arcs[arc]
        return a.cost_interdict / max(a.cost_attack, 1e-9)

    sorted_arcs = sorted(arcs, key=priority)

    x     = {a: 0 for a in arcs}
    budget = B_defender

    for arc in sorted_arcs:
        ci = graph.arcs[arc].cost_interdict
        if ci <= budget:
            x[arc] = 1
            budget -= ci

    return x