"""
model/paper_algorithm.py

Model 1 - Exact replication of the MINMAX algorithm from:
  Nandi, Medal, Vadlamani (2016), "Interdicting attack graphs to protect
  organizations from cyber attacks: A bi-level defender-attacker model",
  Computers & Operations Research, 75, 118-131.

Algorithm flow
--------------
Outer loop alternates between:
  * Sub-problem  (MAXBREACH)      - attacker maximises breach reward
  * Master-problem (MINBREACHPATH) - defender minimises worst-case loss

Convergence when UB - LB <= epsilon.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from model.attack_graph import AttackGraph



# Sub-problem: MAXBREACH (attacker)


def _solve_maxbreach(
    graph: AttackGraph,
    B_attacker: float,
    interdicted: Dict[Tuple[int, int], int],
    solver_msg: bool = False,
) -> Tuple[float, List[Tuple[int, int]], List[List[Tuple[int, int]]]]:
    """
    Solve the attacker sub-problem (MAXBREACH) for a given interdiction plan.

    Uses the disaggregated formulation MaxBreachD (Section 3 of the paper)
    which avoids big-M by disaggregating flows per goal node.

    Parameters
    ----------
    graph       : AttackGraph instance
    B_attacker  : attacker budget
    interdicted : mapping (i,j) -> 1 if arc is interdicted, else 0
    solver_msg  : whether to show Gurobi output

    Returns
    -------
    obj_val  : optimal attacker reward
    used_arcs: list of arcs used (w_ij = 1)
    paths    : list of attack paths (each path is an ordered list of arcs)
    """
    # Free arcs: arcs NOT interdicted
    free = [(i, j) for (i, j) in graph.arcs if interdicted.get((i, j), 0) == 0]

    goal_nodes = [n for n, node in graph.nodes.items() if node.reward > 0]

    m = gp.Model("MaxBreachD")
    m.Params.OutputFlag = 1 if solver_msg else 0

    # w_ij = 1 if arc (i,j) is used in at least one attack path
    w = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}") for (i, j) in free}

    # y^t_ij = 1 if goal node t is attacked via arc (i,j)
    y = {
        (t, i, j): m.addVar(vtype=GRB.BINARY, name=f"y_{t}_{i}_{j}")
        for t in goal_nodes
        for (i, j) in free
    }

    # z_t = 1 if goal node t is breached
    z = {t: m.addVar(vtype=GRB.BINARY, name=f"z_{t}") for t in goal_nodes}

    # Objective: maximise total reward
    m.setObjective(
        gp.quicksum(graph.nodes[t].reward * z[t] for t in goal_nodes),
        GRB.MAXIMIZE,
    )

    # Constraint: z_t <= 1  (each goal at most once) - enforced by binary
    # Constraint (4b): z_t = sum of y^t_it for incoming arcs of t
    for t in goal_nodes:
        incoming_t = [(i, j) for (i, j) in free if j == t]
        m.addConstr(
            z[t] == gp.quicksum(y[t, i, j] for (i, j) in incoming_t),
            name=f"breach_{t}",
        )

    # Constraint (4c): flow balance for goal node t at non-goal intermediate nodes
    non_goal = [n for n in graph.nodes if n not in goal_nodes]
    for t in goal_nodes:
        for i in non_goal:
            if i == 0:
                continue  # vulnerability node: source, skip balance
            out_it = [(i, j) for (i, j) in free if i == i and (i, j) in free and
                      (i, j)[0] == i]  # arcs leaving i
            in_it  = [(k, l) for (k, l) in free if l == i]
            m.addConstr(
                gp.quicksum(y[t, i, j] for (i2, j) in free if i2 == i)
                - gp.quicksum(y[t, k, i] for (k, l) in free if l == i)
                == 0,
                name=f"flow_{t}_{i}",
            )

    # Constraint (4e): y^t_ij <= w_ij
    for t in goal_nodes:
        for (i, j) in free:
            m.addConstr(y[t, i, j] <= w[i, j], name=f"link_{t}_{i}_{j}")

    # Constraint (4f): w_ij <= 1 - x_ij  (already handled: only free arcs included)

    # Constraint (4g): attacker budget
    m.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_attack * w[i, j] for (i, j) in free)
        <= B_attacker,
        name="budget_attacker",
    )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        m.dispose()
        return 0.0, [], []

    obj_val = m.ObjVal
    used_arcs = [(i, j) for (i, j) in free if w[i, j].X > 0.5]

    # Extract one path per breached goal node
    paths: List[List[Tuple[int, int]]] = []
    for t in goal_nodes:
        if z[t].X < 0.5:
            continue
        path: List[Tuple[int, int]] = []
        current = t
        visited = set()
        while True:
            pred = [
                (k, current)
                for (k, l) in free
                if l == current and y[t, k, current].X > 0.5 and (k, current) not in visited
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
    return obj_val, used_arcs, paths



# Master-problem: MINBREACHPATH (defender)


def _solve_minbreachpath(
    graph: AttackGraph,
    B_defender: float,
    attack_paths: List[List[Tuple[int, int]]],
    path_rewards: List[float],
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], int]]:
    """
    Solve the defender master-problem (MINBREACHPATH).

    Formulation (8) from the paper - path-based reformulation.

    Parameters
    ----------
    graph        : AttackGraph instance
    B_defender   : defender budget
    attack_paths : all attack paths collected so far (each is a list of arcs)
    path_rewards : reward associated with each path group (one per iteration)
    solver_msg   : Gurobi output flag

    Returns
    -------
    lb          : lower bound (objective value)
    interdict   : optimal interdiction plan {(i,j): 0/1}
    """
    all_arcs = list(graph.arcs.keys())

    # Group paths by iteration (each element of attack_paths is one group)
    m = gp.Model("MinBreachPath")
    m.Params.OutputFlag = 1 if solver_msg else 0

    # x_ij = 1 if arc (i,j) is interdicted
    x = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for (i, j) in all_arcs}

    # u_p = 1 if path p is removed (all its arcs are interdicted)
    # We create one u_p per path
    all_flat_paths: List[List[Tuple[int, int]]] = []
    for group in attack_paths:
        all_flat_paths.append(group)

    u = [m.addVar(lb=0.0, ub=1.0, name=f"u_{p}") for p in range(len(all_flat_paths))]

    # eta_k = lower bound contribution of attack group k
    # Each iteration contributes one "attack" with a reward
    # We aggregate: one eta per path (reward already computed per path)
    eta = m.addVar(lb=0.0, name="eta")

    m.setObjective(eta, GRB.MINIMIZE)

    # Constraint (8b): eta >= sum_p reward_p * (1 - u_p) for all iterations
    # Since each path has its own reward, and paths come from iterations:
    for idx, path in enumerate(all_flat_paths):
        rew = path_rewards[idx]
        m.addConstr(
            eta >= rew * (1 - u[idx]),
            name=f"obj_cut_{idx}",
        )

    # Constraint (8c): u_p <= x_ij for each arc on path p
    for idx, path in enumerate(all_flat_paths):
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



# Main entry point


def run_paper_algorithm(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    epsilon: float = 1e-6,
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], int], float, int]:
    """
    Run the MINMAX exact algorithm (Algorithm 4.3) from the paper.

    Parameters
    ----------
    graph      : AttackGraph instance
    B_defender : defender budget
    B_attacker : attacker budget
    epsilon    : convergence tolerance
    solver_msg : show Gurobi solver output

    Returns
    -------
    breach_loss  : optimal breach loss (= UB at convergence)
    interdict    : optimal interdiction plan {(i,j): 0/1}
    runtime      : wall-clock time in seconds
    iterations   : number of MINMAX iterations
    """
    t0 = time.time()

    UB = float("inf")
    LB = 0.0
    best_interdict: Dict[Tuple[int, int], int] = {arc: 0 for arc in graph.arcs}

    # Initial interdiction plan: no arcs interdicted
    current_interdict: Dict[Tuple[int, int], int] = {arc: 0 for arc in graph.arcs}

    all_paths: List[List[Tuple[int, int]]] = []
    path_rewards: List[float] = []

    iteration = 0

    while True:
        iteration += 1

        # Step 2: Solve sub-problem (attacker)
        ub_val, used_arcs, paths = _solve_maxbreach(
            graph, B_attacker, current_interdict, solver_msg
        )

        # Step 3: Update UB
        if ub_val < UB:
            UB = ub_val
            best_interdict = dict(current_interdict)

        # Step 4: Convergence check
        if UB - LB <= epsilon:
            break

        # Step 5: Add attack paths to master problem
        for path in paths:
            all_paths.append(path)
            path_rewards.append(ub_val / max(len(paths), 1))

        if not all_paths:
            break

        # Step 6: Solve master problem (defender)
        lb_val, current_interdict = _solve_minbreachpath(
            graph, B_defender, all_paths, path_rewards, solver_msg
        )

        if lb_val > LB:
            LB = lb_val

        if UB - LB <= epsilon:
            break

        if iteration > 500:
            break  # Safety cap

    runtime = time.time() - t0
    return UB, best_interdict, runtime, iteration
