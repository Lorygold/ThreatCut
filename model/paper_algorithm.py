"""
model/paper_algorithm.py

Model 1 – MINMAX exact algorithm (Gurobi, free pip licence).

Faithful reimplementation of Algorithm 4.3 from:
  Nandi, Medal, Vadlamani (2016), Computers & Operations Research 75, 118-131.

The algorithm alternates between:
  Sub-problem   MAXBREACH  (attacker)   – maximises total breach reward
  Master-problem MINBREACHPATH (defender) – minimises worst-case loss

Both models stay well within the 2000-variable / 2000-constraint limit of
the free pip licence when using the instance produced by build_paper_instance()
in main.py (n=38 nodes, 4 levels, ~134 arcs).

Variable counts per solve (worst case for this instance):
  Outer (MinBreachPath): ~200 vars / ~200 constraints  (grows with iterations)
  Inner (MaxBreachD):    ~1500 vars / ~1560 constraints
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from model.attack_graph import AttackGraph



# Sub-problem: MaxBreachD  (attacker, disaggregated – no big-M)


def _solve_maxbreach(
    graph: AttackGraph,
    B_attacker: float,
    interdicted: Dict[Tuple[int, int], int],
    solver_msg: bool = False,
) -> Tuple[float, List[List[Tuple[int, int]]]]:
    """
    Solve the attacker sub-problem (MaxBreachD) for a given interdiction plan.

    Uses the disaggregated formulation (Section 3 of the paper):
    flow variables y^t_{ij} track attacks aimed at each specific goal node t,
    eliminating the need for a big-M constant.

    Returns
    -------
    obj_val : optimal attacker reward
    paths   : list of attack paths (one list of arcs per breached goal node)
    """
    free = [(i, j) for (i, j) in graph.arcs if interdicted.get((i, j), 0) == 0]
    goal_nodes = [n for n, nd in graph.nodes.items() if nd.reward > 0]

    if not free or not goal_nodes:
        return 0.0, []

    m = gp.Model("MaxBreachD")
    m.Params.OutputFlag = 1 if solver_msg else 0

    # w[i,j] = 1 if arc (i,j) is used in at least one attack path
    w = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}") for (i, j) in free}

    # y[t,i,j] = 1 if goal t is attacked via arc (i,j)
    y = {
        (t, i, j): m.addVar(vtype=GRB.BINARY, name=f"y_{t}_{i}_{j}")
        for t in goal_nodes
        for (i, j) in free
    }

    # z[t] = 1 if goal node t is breached
    z = {t: m.addVar(vtype=GRB.BINARY, name=f"z_{t}") for t in goal_nodes}

    # Maximise total breach reward
    m.setObjective(
        gp.quicksum(graph.nodes[t].reward * z[t] for t in goal_nodes),
        GRB.MAXIMIZE,
    )

    # Constraint (4b): z[t] = sum of y[t,i,t] for incoming arcs of goal t
    for t in goal_nodes:
        m.addConstr(
            z[t] == gp.quicksum(y[t, i, j] for (i, j) in free if j == t),
            name=f"breach_{t}",
        )

    # Constraint (4c)+(4d): flow balance – attacks for goal t must not stop
    # at intermediate nodes (non-source, non-goal)
    non_goal_non_src = [n for n in graph.nodes if n not in goal_nodes and n != 0]
    for t in goal_nodes:
        for node in non_goal_non_src:
            m.addConstr(
                gp.quicksum(y[t, node, j] for (_, j) in free if _ == node)
                - gp.quicksum(y[t, k, node] for (k, _) in free if _ == node)
                == 0,
                name=f"flow_{t}_{node}",
            )

    # Constraint (4e): y[t,i,j] <= w[i,j]
    for t in goal_nodes:
        for (i, j) in free:
            m.addConstr(y[t, i, j] <= w[i, j], name=f"link_{t}_{i}_{j}")

    # Constraint (4g): attacker budget
    m.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_attack * w[i, j] for (i, j) in free)
        <= B_attacker,
        name="budget_attacker",
    )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        m.dispose()
        return 0.0, []

    obj_val = m.ObjVal

    # Extract one attack path per breached goal node
    paths: List[List[Tuple[int, int]]] = []
    for t in goal_nodes:
        if z[t].X < 0.5:
            continue
        path: List[Tuple[int, int]] = []
        current = t
        visited: set = set()
        for _ in range(len(free) + 1):
            pred = [
                (k, current)
                for (k, l) in free
                if l == current
                and (k, current) not in visited
                and y[t, k, current].X > 0.5
            ]
            if not pred:
                break
            arc = pred[0]
            path.insert(0, arc)
            visited.add(arc)
            current = arc[0]
            if current == 0:
                break
        if path:
            paths.append(path)

    m.dispose()
    return obj_val, paths



# Master-problem: MinBreachPath  (defender, path-based – eq. 8 of the paper)


def _solve_minbreachpath(
    graph: AttackGraph,
    B_defender: float,
    all_paths: List[Tuple[List[Tuple[int, int]], float]],
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], int]]:
    """
    Solve the path-based defender master-problem (MinBreachPath, eq. 8).

    Each entry in all_paths is a (path, reward) pair collected across iterations.
    The model grows by one variable u_p and a few constraints per new path.

    Returns
    -------
    lb        : lower bound
    interdict : optimal interdiction plan {(i,j): 0/1}
    """
    all_arcs = list(graph.arcs.keys())

    m = gp.Model("MinBreachPath")
    m.Params.OutputFlag = 1 if solver_msg else 0

    # x[i,j] = 1 if arc interdicted by defender
    x = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for (i, j) in all_arcs}

    # u[p] in [0,1]: 1 if path p is fully covered (all arcs interdicted)
    u = [m.addVar(lb=0.0, ub=1.0, name=f"u_{p}") for p in range(len(all_paths))]

    eta = m.addVar(lb=0.0, name="eta")

    m.setObjective(eta, GRB.MINIMIZE)

    # Constraint (8b): eta >= reward_p * (1 - u_p)  for each path p
    for idx, (_, rew) in enumerate(all_paths):
        m.addConstr(eta >= rew * (1 - u[idx]), name=f"obj_cut_{idx}")

    # Constraint (8c): u_p <= x[i,j] for each arc on path p
    for idx, (path, _) in enumerate(all_paths):
        for (i, j) in path:
            if (i, j) in x:
                m.addConstr(u[idx] <= x[i, j], name=f"path_arc_{idx}_{i}_{j}")

    # Constraint (8d): defender budget
    m.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_interdict * x[i, j] for (i, j) in all_arcs)
        <= B_defender,
        name="budget_defender",
    )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        m.dispose()
        return 0.0, {arc: 0 for arc in all_arcs}

    lb = m.ObjVal
    interdict = {(i, j): int(round(x[i, j].X)) for (i, j) in all_arcs}
    m.dispose()
    return lb, interdict



# Entry point


def run_paper_algorithm(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    epsilon: float = 1e-4,
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], int], float, int]:
    """
    Run the MINMAX exact algorithm (Algorithm 4.3) from the paper.

    Returns
    -------
    breach_loss : optimal breach loss
    interdict   : interdiction plan {(i,j): 0/1}
    runtime     : wall-clock seconds
    iterations  : number of outer-loop iterations
    """
    t0 = time.time()

    UB = float("inf")
    LB = 0.0
    best_interdict: Dict[Tuple[int, int], int] = {arc: 0 for arc in graph.arcs}
    current_interdict = dict(best_interdict)

    all_paths: List[Tuple[List[Tuple[int, int]], float]] = []
    iteration = 0

    while True:
        iteration += 1

        # Solve attacker sub-problem
        ub_val, paths = _solve_maxbreach(
            graph, B_attacker, current_interdict, solver_msg
        )

        if ub_val < UB:
            UB = ub_val
            best_interdict = dict(current_interdict)

        if UB - LB <= epsilon:
            break

        # Add all attack paths found in this iteration to the master problem
        per_path_rew = ub_val / max(len(paths), 1)
        for path in paths:
            all_paths.append((path, per_path_rew))

        if not all_paths:
            break

        # Solve defender master problem
        lb_val, current_interdict = _solve_minbreachpath(
            graph, B_defender, all_paths, solver_msg
        )
        if lb_val > LB:
            LB = lb_val

        if UB - LB <= epsilon:
            break

        if iteration >= 200:
            break

    return UB, best_interdict, time.time() - t0, iteration