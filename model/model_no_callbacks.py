"""
model/model_no_callbacks.py

Model 3 – Iterative Benders decomposition without callbacks (Gurobi).

Faithful refactoring of new_Gurobi_no_callbacks.py.

The outer and inner MIPs alternate in a plain Python while-loop:
  1. Solve outer MIP  → current interdiction plan y.
  2. Identify free arcs (y_ij ≈ 0).
  3. Solve inner attacker problem over free arcs.
  4. Add a Benders optimality cut to the outer MIP.
  5. Repeat until UB - LB ≤ epsilon.

Unlike Model 2, no Gurobi callback is used; cuts are added via
model.addConstr() between successive calls to model.optimize().

Variable counts (n=38 nodes, ~134 arcs):
  Outer MIP: y(134) + w(~424) + z(1) = ~559 vars / ~425 constraints + cuts
  Inner MIP: x(<=134) + u(<=134) + v(1) = <=269 vars / <=411 constraints
  Both well within the 2000-var / 2000-constraint free-licence limit.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from model.attack_graph import AttackGraph


def run_no_callbacks(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    epsilon: float = 1e-4,
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], int], float, int]:
    """
    Solve MINMAXBREACH via iterative Benders cuts (no callbacks).

    The outer MIP is kept in memory and re-optimised after each new cut.
    The inner attacker MIP is rebuilt from scratch at each iteration
    (only over the free arcs of the current outer solution).

    Returns
    -------
    breach_loss : optimal breach loss
    interdict   : interdiction plan {(i,j): 0/1}
    runtime     : wall-clock seconds
    iterations  : number of Benders iterations
    """
    arcs = list(graph.arcs.keys())
    trew: float = sum(graph.nodes[i].reward for i in graph.nodes)

    
    # Build outer MIP (persists across iterations)
    
    outer = gp.Model("Outer_NoCB")
    outer.Params.OutputFlag = 1 if solver_msg else 0

    y = {(i, j): outer.addVar(vtype=GRB.BINARY,     name=f"y_{i}_{j}") for (i, j) in arcs}
    w = {
        (i, j, k): outer.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}_{k}")
        for (i, j) in arcs
        for (l, k) in arcs
        if j == l
    }
    z = outer.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    outer.setObjective(z, GRB.MINIMIZE)

    outer.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_interdict * y[i, j] for i, j in arcs)
        <= B_defender,
        name="budget_defender",
    )

    for i, j in arcs:
        for l, k in arcs:
            if l == j:
                outer.addConstr(
                    w[i, j, k] >= y[i, j] + y[j, k] - 1,
                    name=f"w_link_{i}_{j}_{k}",
                )

    
    # Iterative Benders loop
    
    t0 = time.time()

    outer.optimize()
    LB = z.X
    UB = trew  # pessimistic upper bound before first inner solve

    free = [(i, j) for (i, j) in arcs if y[i, j].X < 1e-8]
    iteration = 0
    last_interdict = {(i, j): int(round(y[i, j].X)) for (i, j) in arcs}

    while LB < UB - epsilon:
        iteration += 1

        
        # Inner attacker problem
        
        inner = gp.Model("Inner_NoCB")
        inner.Params.OutputFlag = 0

        x = {(i, j): inner.addVar(vtype=GRB.BINARY,    name=f"x_{i}_{j}") for (i, j) in free}
        u = {(i, j): inner.addVar(vtype=GRB.CONTINUOUS, name=f"u_{i}_{j}") for (i, j) in free}
        v = inner.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")

        inner.setObjective(v, GRB.MAXIMIZE)

        inner.addConstr(
            v == gp.quicksum(u[i, j] for (i, j) in free if i == 0),
            name="obj_def",
        )
        inner.addConstr(
            gp.quicksum(graph.arcs[i, j].cost_attack * x[i, j] for (i, j) in free)
            <= B_attacker,
            name="budget_attacker",
        )
        for node in graph.nodes:
            inner.addConstr(
                gp.quicksum(x[k, l] for (k, l) in free if l == node) <= 1,
                name=f"in_degree_{node}",
            )
        for i, j in free:
            inner.addConstr(u[i, j] <= trew * x[i, j], name=f"flow_ub_{i}_{j}")
            if graph.nodes[j].reward <= 0:
                inner.addConstr(
                    u[i, j] <= gp.quicksum(u[k, l] for (k, l) in free if k == j),
                    name=f"flow_prop_{i}_{j}",
                )
            else:
                inner.addConstr(
                    u[i, j] <= graph.nodes[j].reward * x[i, j],
                    name=f"goal_reward_{i}_{j}",
                )

        inner.optimize()

        v_star = v.X
        u_star = {(i, j): u[i, j].X for (i, j) in free}

        if v_star < UB:
            UB = v_star

        inner.dispose()

        
        # Add Benders cut to outer MIP and re-solve
        
        outer.addConstr(
            z >= v_star
            - gp.quicksum(u_star[i, j] * y[i, j] for (i, j) in free)
            + gp.quicksum(
                u_star[j, k] * w[i, j, k]
                for (i, j) in free
                for (l, k) in free
                if j == l
            ),
            name=f"benders_cut_{iteration}",
        )

        outer.optimize()
        LB = z.X
        last_interdict = {(i, j): int(round(y[i, j].X)) for (i, j) in arcs}
        free = [(i, j) for (i, j) in arcs if y[i, j].X < 1e-8]

        if iteration >= 200:
            break

    runtime = time.time() - t0
    outer.dispose()
    return LB, last_interdict, runtime, iteration
