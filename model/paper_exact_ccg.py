"""
model/paper_exact_ccg.py

Implementation of the MINMAX algorithm from:

    Nandi, Medal, Vadlamani (2016)
    "Interdicting Attack Graphs to Protect Organizations from Cyber Attacks"
    Computers & Operations Research, Vol. 75, pp. 118-131

This module implements the EXACT algorithm (EM) described in Section 4:

  Outer level  → MINBREACHPATH master problem (Section 4.5.1)
                 Path-based formulation — much more efficient than the
                 node-based MINBREACHNODE formulation (Section 4.2).
  Inner level  → MAXBREACHBM sub-problem (Section 3.1)
                 Big-M formulation solved as a MIP.

The algorithm alternates between master and sub-problem:
  1. Sub-problem MAXBREACHBM(x_hat) → upper bound + attack plan A_bar
  2. Extract paths from A_bar (Lemma 1: attack is a tree → at most |N_T| paths)
  3. Add new path variables u_p and constraints to MINBREACHPATH
  4. Solve MINBREACHPATH → lower bound + new interdiction plan x_hat
  5. Repeat until UB - LB <= epsilon

Enhancements from Section 4.5 included:
  - Path-based master (S)          — always active
  - Multiple sub-problem solutions (Ms) — adds top-k solutions per iteration
  - Trust region cuts (TR)         — stabilises first 20 iterations

Solver: Gurobi (via gurobipy). The paper used Gurobi as well.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.attack_graph import AttackGraph


# Result dataclass

@dataclass
class PaperCCGResult:
    x_optimal: Dict[Tuple[int, int], int]   # optimal interdiction plan
    breach_loss: float                        # optimal worst-case breach loss
    n_iterations: int
    lower_bounds: List[float] = field(default_factory=list)
    upper_bounds: List[float] = field(default_factory=list)
    solve_time_s: float = 0.0
    n_paths_added: int = 0


# Sub-problem: MAXBREACHBM (Section 3.1)

def _solve_maxbreachbm(
    graph: AttackGraph,
    x_hat: Dict[Tuple[int, int], int],
    B_attacker: float,
    solver_msg: bool = False,
    n_solutions: int = 1,
) -> Tuple[float, List[List[int]]]:
    """
    Solve MAXBREACHBM: attacker maximises total reward given interdiction x_hat.

    Variables (Section 3.1):
        w_ij ∈ {0,1}  : 1 if arc (i,j) is used in at least one attack path
        y_ij ≥ 0      : number of goal nodes attacked through arc (i,j)
        z_t ∈ {0,1}   : 1 if goal node t is breached  (relaxed — integer by proof)

    Returns
    -------
    (upper_bound, list_of_attack_paths)
        upper_bound   : optimal attacker profit
        attack_paths  : list of paths [list of node ids] used in the attack
    """
    # Arcs not interdicted by the defender
    free_arcs = [(i, j) for (i, j) in graph.arcs if x_hat.get((i, j), 0) == 0]
    goal_ids  = [g.id for g in graph.goal_nodes]
    M         = len(goal_ids)  # big-M = |N_T| (Section 3.1, constraint 2d)

    if not free_arcs:
        return 0.0, []

    inner = gp.Model("MAXBREACHBM")
    inner.Params.OutputFlag = 1 if solver_msg else 0
    inner.Params.PoolSearchMode = 2 if n_solutions > 1 else 0
    inner.Params.PoolSolutions = n_solutions

    # Decision variables
    w = {(i, j): inner.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}")
         for (i, j) in free_arcs}
    y = {(i, j): inner.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"y_{i}_{j}")
         for (i, j) in free_arcs}
    z = {t: inner.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"z_{t}")
         for t in goal_ids}

    # Objective: maximise total reward (2a)
    inner.setObjective(
        gp.quicksum(graph.nodes[t].reward * z[t] for t in goal_ids),
        GRB.MAXIMIZE
    )

    # (2b) z_t = 0 unless at least one arc enters goal t is active
    for t in goal_ids:
        in_arcs = [(i, j) for (i, j) in free_arcs if j == t]
        if in_arcs:
            inner.addConstr(
                z[t] <= gp.quicksum(y[i, j] for (i, j) in in_arcs),
                name=f"goal_reach_{t}"
            )
        else:
            inner.addConstr(z[t] == 0, name=f"goal_unreachable_{t}")
        inner.addConstr(z[t] <= 1, name=f"goal_once_{t}")  # (2g)

    # (2c) flow balance at transition nodes: attacks do not stop mid-path
    transition_ids = {
        n.id for n in graph.nodes.values()
        if n.level > 0 and n.id not in set(goal_ids)
    }
    for node_id in transition_ids:
        in_a  = [(i, j) for (i, j) in free_arcs if j == node_id]
        out_a = [(i, j) for (i, j) in free_arcs if i == node_id]
        if in_a and out_a:
            inner.addConstr(
                gp.quicksum(y[i, j] for (i, j) in in_a) ==
                gp.quicksum(y[i, j] for (i, j) in out_a),
                name=f"flow_{node_id}"
            )

    # (2d) y_ij <= M * w_ij
    for (i, j) in free_arcs:
        inner.addConstr(y[i, j] <= M * w[i, j], name=f"bigM_{i}_{j}")

    # (2e) arc not used if interdicted — already filtered into free_arcs

    # (2f) attacker budget
    inner.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_attack * w[i, j]
                    for (i, j) in free_arcs) <= B_attacker,
        name="attacker_budget"
    )

    inner.optimize()

    if inner.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        inner.dispose()
        return 0.0, []

    obj = inner.ObjVal

    # Extract attack paths from the w solution
    all_paths: List[List[int]] = []
    for sol_idx in range(inner.SolCount):
        inner.Params.SolutionNumber = sol_idx
        used = [(i, j) for (i, j) in free_arcs if inner.getVarByName(f"w_{i}_{j}").Xn > 0.5]
        paths = _extract_paths_from_arcs(graph, used)
        for p in paths:
            if p not in all_paths:
                all_paths.append(p)

    inner.dispose()
    return obj, all_paths


def _extract_paths_from_arcs(
    graph: AttackGraph,
    used_arcs: List[Tuple[int, int]],
) -> List[List[int]]:
    """
    Reconstruct source-to-goal paths from the set of arcs used in the attack.
    Per Lemma 1 of the paper, an attack is a tree, so each goal node has
    exactly one path from a source.
    """
    if not used_arcs:
        return []

    g_sub = nx.DiGraph()
    g_sub.add_edges_from(used_arcs)

    goal_ids = {n.id for n in graph.goal_nodes}
    source_ids = {n.id for n in graph.source_nodes}

    paths: List[List[int]] = []
    for src in source_ids:
        if src not in g_sub:
            continue
        for goal in goal_ids:
            if goal not in g_sub:
                continue
            try:
                for p in nx.all_simple_paths(g_sub, src, goal):
                    paths.append(p)
            except nx.NetworkXNoPath:
                pass
    return paths


# Master problem: MINBREACHPATH (Section 4.5.1)

def _solve_minbreachpath(
    graph: AttackGraph,
    paths: List[List[int]],
    B_defender: float,
    B_attacker: float,
    trust_region: Optional[Dict[Tuple[int, int], int]] = None,
    tr_limit: float = 0.33,
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], int]]:
    """
    Solve MINBREACHPATH master problem (Section 4.5.1).

    Variables:
        x_ij ∈ {0,1} : interdiction plan
        u_p  ∈ [0,1] : 1 if path p is blocked (relaxed — binary by proof)
        η    ≥ 0     : worst-case total loss

    Constraints per path p (8b): η >= reward(p) * (1 - u_p)
    Constraints (8c): u_p <= sum_{(i,j) in p} x_ij  (blocked if any arc on p is cut)

    Returns
    -------
    (lower_bound, x_solution)
    """
    arcs = list(graph.arcs.keys())

    master = gp.Model("MINBREACHPATH")
    master.Params.OutputFlag = 1 if solver_msg else 0

    # Variables
    x = {(i, j): master.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
         for (i, j) in arcs}
    eta = master.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="eta")

    # Objective (8a): minimise worst-case loss
    master.setObjective(eta, GRB.MINIMIZE)

    # Defender budget (8d)
    master.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_interdict * x[i, j]
                    for (i, j) in arcs) <= B_defender,
        name="defender_budget"
    )

    # Path constraints: add u_p per path and link it to x
    u = {}
    for idx, path in enumerate(paths):
        goal_id = path[-1]
        if graph.nodes[goal_id].level != graph._num_levels:
            continue
        if graph.path_cost_attack(path) > B_attacker:
            # Path not affordable by attacker — safe to skip
            continue
        reward = graph.nodes[goal_id].reward
        path_arc_list = graph.path_arcs(path)

        u_p = master.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"u_{idx}")
        u[idx] = u_p

        # (8b): η >= reward * (1 - u_p)
        master.addConstr(eta >= reward * (1 - u_p), name=f"loss_cut_{idx}")

        # (8c): u_p <= sum_{(i,j) in path} x_ij
        master.addConstr(
            u_p <= gp.quicksum(x[i, j] for (i, j) in path_arc_list),
            name=f"block_path_{idx}"
        )

    # Trust region cut (Section 4.5.3): limit change from previous solution
    if trust_region is not None:
        X_prev = [(i, j) for (i, j) in arcs if trust_region.get((i, j), 0) == 1]
        X_not  = [(i, j) for (i, j) in arcs if trust_region.get((i, j), 0) == 0]
        if X_prev:
            max_change = max(1, int(tr_limit * 2 * len(X_prev)))
            master.addConstr(
                gp.quicksum(1 - x[i, j] for (i, j) in X_prev) +
                gp.quicksum(x[i, j] for (i, j) in X_not) <= max_change,
                name="trust_region"
            )

    master.optimize()

    if master.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        master.dispose()
        return 0.0, {(i, j): 0 for (i, j) in arcs}

    lb = master.ObjVal
    x_sol = {(i, j): int(round(x[i, j].X or 0)) for (i, j) in arcs}

    master.dispose()
    return lb, x_sol


# Main algorithm: MINMAX (Algorithm 1 + enhancements Section 4.5)

def run_paper_ccg(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    epsilon: float = 1e-4,
    max_iter: int = 500,
    use_multiple_solutions: bool = True,   # enhancement Ms (Section 4.5.2)
    use_trust_region: bool = True,          # enhancement TR (Section 4.5.3)
    n_tr_iterations: int = 20,             # number of TR iterations
    solver_msg: bool = False,
) -> PaperCCGResult:
    """
    Solve the MINMAXBREACH problem using the exact MINMAX algorithm
    (Algorithm 1) with path-based master and optional enhancements.

    The algorithm:
      Iteration 1: x_hat = 0 (no interdiction).
      Step 2: Solve MAXBREACHBM(x_hat) → UB, attack paths.
      Step 3: Update UB. If UB - LB <= epsilon, stop.
      Step 4: Add new paths to master. Solve MINBREACHPATH → LB, new x_hat.
      Step 5: If UB - LB <= epsilon, stop. Else k = k+1, goto Step 2.

    Parameters
    ----------
    graph        : AttackGraph
    B_defender   : float
    B_attacker   : float
    epsilon      : float  convergence tolerance
    max_iter     : int    safety cap
    use_multiple_solutions : bool  add multiple sub-problem solutions (Ms)
    use_trust_region       : bool  add trust region cuts for first n_tr_iterations
    n_tr_iterations        : int   how many TR iterations to apply
    solver_msg   : bool

    Returns
    -------
    PaperCCGResult
    """
    t0 = time.perf_counter()

    arcs = list(graph.arcs.keys())

    # Step 1: initialise
    ub = float("inf")
    lb = 0.0
    x_hat: Dict[Tuple[int, int], int] = {a: 0 for a in arcs}
    best_x: Dict[Tuple[int, int], int] = {a: 0 for a in arcs}
    paths: List[List[int]] = []
    seen_paths: Set[str] = set()

    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    # How many sub-problem solutions to request (Ms enhancement)
    # Paper uses 33% of available solutions; we cap at a reasonable number.
    n_sol = 10 if use_multiple_solutions else 1

    prev_x: Optional[Dict[Tuple[int, int], int]] = None

    for k in range(1, max_iter + 1):

        # Step 2: solve sub-problem
        new_ub, attack_paths = _solve_maxbreachbm(
            graph, x_hat, B_attacker,
            solver_msg=solver_msg,
            n_solutions=n_sol,
        )

        # Step 3: update upper bound
        if new_ub < ub:
            ub = new_ub
            best_x = dict(x_hat)

        upper_bounds.append(ub)

        if ub - lb <= epsilon:
            lower_bounds.append(lb)
            break

        # Step 4: add new paths to the master
        added = False
        for path in attack_paths:
            key = str(path)
            if key not in seen_paths:
                seen_paths.add(key)
                paths.append(path)
                added = True

        if not added:
            # Sub-problem produced no new paths - convergence by exhaustion
            lower_bounds.append(lb)
            break

        # Decide whether to apply trust region
        tr = prev_x if (use_trust_region and k <= n_tr_iterations) else None

        # Step 5: solve master
        lb, x_hat = _solve_minbreachpath(
            graph, paths, B_defender, B_attacker,
            trust_region=tr,
            solver_msg=solver_msg,
        )
        prev_x = dict(x_hat)
        lower_bounds.append(lb)

        if ub - lb <= epsilon:
            best_x = dict(x_hat)
            break

    n_iter = len(upper_bounds)
    gap = (ub - lb) / max(ub, 1e-9) if ub > 1e-9 else 0.0

    return PaperCCGResult(
        x_optimal=best_x,
        breach_loss=ub,
        n_iterations=n_iter,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        solve_time_s=time.perf_counter() - t0,
        n_paths_added=len(paths),
    )