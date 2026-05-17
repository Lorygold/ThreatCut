"""
model/new_no_callbacks.py

Gurobi-based implementation of the bilevel defender-attacker interdiction
model using an EXPLICIT SEQUENTIAL BENDERS LOOP (no callbacks).

This is the "classic" Benders decomposition approach:
  1. Solve the outer (defender) problem → get interdiction plan y, lower bound LB
  2. Identify free arcs F = {(i,j) : y_{ij} = 0}
  3. Solve the inner (attacker) problem on F → get attacker profit v, upper bound UB
  4. Add a Benders cut to the outer model:
         z >= v - sum u_{ij}*y_{ij} + sum u_{jk}*w_{ijk}
  5. Re-solve the outer problem → new LB
  6. Repeat until LB >= UB - epsilon

The outer model uses auxiliary variables w_{ijk} to handle the correction
factor when two consecutive arcs (i,j) and (j,k) are both interdicted,
avoiding double-counting in the Benders cut.

Compared to new_callbacks.py, this version is simpler to understand and
debug, but less efficient: cuts are only added at the root of the B&B tree
(after a full re-solve of the outer problem) rather than throughout the tree.

For large instances, the callback version (new_callbacks.py) should be
significantly faster.

Implementation suggested by Roberto Montemanni.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from model.attack_graph import AttackGraph


def _build_inner_problem(
    graph: AttackGraph,
    free_arcs: List[Tuple[int, int]],
    B_attacker: float,
    M: float,
    L: int,
    W: int,
) -> Tuple[gp.Model, Dict, Dict, gp.Var]:
    """
    Build the attacker's inner MIP for the given set of free (non-interdicted) arcs.

    The inner problem maximises the total profit the attacker can collect
    by choosing an arborescence rooted at node 0 (the source) subject to
    the attacker's budget constraint.

    The profit propagation model works as follows:
      - u[i,j] represents the profit "carried" along arc (i,j).
      - For arcs entering goal nodes: u[i,j] <= r_j * x[i,j]  (reward capped)
      - For arcs entering intermediate nodes: u[i,j] <= sum_{(j,k)} u[j,k]
        (profit flows forward; a node can only pass on what it collects ahead)
      - The total profit v = sum_{(0,j)} u[0,j]  (collected at root)

    Parameters
    ----------
    graph      : AttackGraph
    free_arcs  : list of (i,j) tuples - arcs not interdicted by the defender
    B_attacker : float - attacker's budget
    M          : float - big-M constant = sum of all goal node rewards
    L          : int   - number of levels (for valid inequality; 0 to disable)
    W          : int   - nodes per level (for valid inequality; 0 to disable)

    Returns
    -------
    (inner_model, x_vars, u_vars, v_var)
    """
    inner = gp.Model("inner")
    inner.Params.OutputFlag = 0

    x = {(i, j): inner.addVar(vtype=GRB.BINARY,     name=f"x_{i}_{j}") for (i, j) in free_arcs}
    u = {(i, j): inner.addVar(vtype=GRB.CONTINUOUS,  name=f"u_{i}_{j}") for (i, j) in free_arcs}
    v = inner.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")

    inner.setObjective(v, GRB.MAXIMIZE)

    # Total profit at root = sum of u on arcs leaving node 0
    inner.addConstr(
        v == gp.quicksum(u[i, j] for (i, j) in free_arcs if i == 0),
        name="root_profit",
    )

    # Attacker budget
    inner.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_attack * x[i, j] for (i, j) in free_arcs)
        <= B_attacker,
        name="attacker_budget",
    )

    # Arborescence: at most one incoming arc per node
    for node_id in graph.nodes:
        in_arcs = [(i, j) for (i, j) in free_arcs if j == node_id]
        if in_arcs:
            inner.addConstr(
                gp.quicksum(x[i, j] for (i, j) in in_arcs) <= 1,
                name=f"arborescence_{node_id}",
            )

    # Profit propagation
    for (i, j) in free_arcs:
        inner.addConstr(u[i, j] <= M * x[i, j], name=f"bigM_{i}_{j}")
        if graph.nodes[j].reward > 0:
            inner.addConstr(
                u[i, j] <= graph.nodes[j].reward * x[i, j],
                name=f"goal_reward_{i}_{j}",
            )
        else:
            out_arcs = [(l, k) for (l, k) in free_arcs if l == j]
            if out_arcs:
                inner.addConstr(
                    u[i, j] <= gp.quicksum(u[j, k] for (j, k) in out_arcs),
                    name=f"flow_fwd_{i}_{j}",
                )

    # Valid inequality: total flow crossing each inter-level cut equals v.
    # Applied once per level, outside the arc loop.
    # Only meaningful for regular L×W graphs (skip when L or W is 0).
    if L > 0 and W > 0:
        for l in range(L):
            idx  = 1 + (l - 1) * W
            idx2 = 1 + l * W
            inner.addConstr(
                gp.quicksum(
                    u[i, j]
                    for i in range(idx,  idx  + W)
                    for j in range(idx2, idx2 + W)
                    if (i, j) in x
                ) == v,
                name=f"level_flow_{l}",
            )

    return inner, x, u, v


def run_new_no_callbacks(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    Lp: int,
    Wp: int,
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], int], int]:
    """
    Solve the bilevel cyber attack interdiction problem using an explicit
    sequential Benders decomposition loop (no Gurobi callbacks).

    Algorithm
    ---------
    Iteration 0:
        Solve the outer problem with no Benders cuts → LB = z*, extract plan y*.
        Set UB = sum of all goal rewards (trivial upper bound).

    Each subsequent iteration:
        1. Build free arc set F = {(i,j) : y*_{ij} = 0}
        2. Solve inner problem on F → attacker profit v*, arc profits u*
        3. Update UB = min(UB, v*)
        4. Add Benders cut to outer:
               z >= v* - sum_{(i,j) in F} u*_{ij} * y_{ij}
                       + sum_{(i,j),(j,k) in F} u*_{jk} * w_{ijk}
        5. Re-solve outer → new LB = z*
        6. Extract new plan y*, update free arc set
        7. Terminate when LB >= UB - epsilon

    Parameters
    ----------
    graph      : AttackGraph
    B_defender : float  Defender's interdiction budget.
    B_attacker : float  Attacker's traversal budget.
    Lp         : int    Number of levels (used for valid inequality).
    Wp         : int    Nodes per level (used for valid inequality).
    solver_msg : bool   Show Gurobi output if True.

    Returns
    -------
    (breach_loss, x_optimal, n_iterations)
        breach_loss  : optimal worst-case attacker profit
        x_optimal    : dict {(i,j): 0 or 1} - optimal interdiction plan
        n_iterations : number of Benders iterations performed
    """
    arcs = list(graph.arcs.keys())
    M    = sum(n.reward for n in graph.nodes.values())

    outer = gp.Model("outer_no_callbacks")
    outer.Params.OutputFlag = 1 if solver_msg else 0

    y = {(i, j): outer.addVar(vtype=GRB.BINARY,     name=f"y_{i}_{j}") for (i, j) in arcs}
    w = {
        (i, j, k): outer.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}_{k}")
        for (i, j) in arcs
        for (l, k) in arcs if j == l
    }
    z = outer.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    outer.setObjective(z, GRB.MINIMIZE)

    outer.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_interdict * y[i, j] for (i, j) in arcs)
        <= B_defender,
        name="defender_budget",
    )

    for (i, j) in arcs:
        for (l, k) in arcs:
            if l == j:
                outer.addConstr(w[i, j, k] >= y[i, j] + y[j, k] - 1, name=f"w_align_{i}_{j}_{k}")
                outer.addConstr(w[i, j, k] <= y[i, j],                 name=f"w_align_ij_{i}_{j}_{k}")
                outer.addConstr(w[i, j, k] <= y[j, k],                 name=f"w_align_jk_{i}_{j}_{k}")

    outer.optimize()

    lb   = z.X
    ub   = M
    free = [(i, j) for (i, j) in arcs if y[i, j].X < 1e-7]

    print(f"  Init  |  LB = {lb:.4f}  |  UB = {ub:.4f}")

    n_iter = 0

    while lb < ub - 1e-7:
        n_iter += 1

        inner, x_vars, u_vars, v_var = _build_inner_problem(
            graph, free, B_attacker, M, Lp, Wp
        )
        inner.optimize()

        attacker_profit = v_var.X
        if attacker_profit < ub:
            ub = attacker_profit

        outer.addConstr(
            z >= attacker_profit
            - gp.quicksum(u_vars[i, j].X * y[i, j] for (i, j) in free)
            + gp.quicksum(
                u_vars[j, k].X * w[i, j, k]
                for (i, j) in free
                for (l, k) in free if j == l
            ),
            name=f"benders_cut_{n_iter}",
        )

        inner.dispose()

        outer.optimize()
        lb   = z.X
        free = [(i, j) for (i, j) in arcs if y[i, j].X < 1e-7]

        print(f"  Iter {n_iter:3d}  |  LB = {lb:.4f}  |  UB = {ub:.4f}")

    print(f"\nConverged after {n_iter} iteration(s).  Breach loss = {ub:.4f}")

    x_optimal = {(i, j): int(round(y[i, j].X or 0)) for (i, j) in arcs}
    outer.dispose()
    return ub, x_optimal, n_iter
