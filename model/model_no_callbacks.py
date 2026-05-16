"""
model/model_no_callbacks.py

Model 3 - Refactored version of new_Gurobi_no_callbacks.py.

Iterative Benders decomposition WITHOUT Gurobi callbacks.
The outer and inner MIPs are solved in alternation in a Python while-loop:
  1. Solve outer model to get current interdiction plan (y).
  2. Identify free arcs (y_ij = 0).
  3. Solve inner attacker problem over free arcs.
  4. Add Benders optimality cut to outer model.
  5. Repeat until UB - LB <= tolerance.

This approach is equivalent to Model 2 but uses the standard
"re-optimize outer" loop instead of Gurobi callbacks.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import gurobipy as gp
from gurobipy import GRB

from model.attack_graph import AttackGraph


def run_no_callbacks(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    epsilon: float = 1e-6,
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], int], float, int]:
    """
    Solve the MINMAXBREACH problem via iterative Benders cuts (no callbacks).

    Parameters
    ----------
    graph      : AttackGraph instance
    B_defender : defender budget
    B_attacker : attacker budget
    epsilon    : optimality gap tolerance
    solver_msg : show Gurobi output for outer/inner models

    Returns
    -------
    breach_loss  : optimal breach loss
    interdict    : interdiction plan {(i,j): 0/1}
    runtime      : wall-clock time in seconds
    iterations   : number of Benders iterations
    """
    arcs = list(graph.arcs.keys())

    # Total reward (big-M proxy for flow upper bound)
    trew: float = sum(graph.nodes[i].reward for i in graph.nodes)

    
    # Build outer model (persisted across iterations)
    
    outer = gp.Model("outer_no_cb")
    outer.Params.OutputFlag = 1 if solver_msg else 0

    y = {
        (i, j): outer.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
        for (i, j) in arcs
    }
    w = {
        (i, j, k): outer.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}_{k}")
        for (i, j) in arcs
        for (l, k) in arcs
        if j == l
    }
    z = outer.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    outer.setObjective(z, GRB.MINIMIZE)

    # Defender budget
    outer.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_interdict * y[i, j] for i, j in arcs)
        <= B_defender,
        name="budget_defender",
    )

    # w linking constraints
    for i, j in arcs:
        for l, k in arcs:
            if l == j:
                outer.addConstr(
                    w[i, j, k] >= y[i, j] + y[j, k] - 1,
                    name=f"w_link_{i}_{j}_{k}",
                )

    
    # Iterative Benders loop
    
    t0 = time.time()

    # Initial solve (no cuts yet)
    outer.optimize()
    LB = z.X
    UB = trew  # pessimistic upper bound

    free = [(i, j) for (i, j) in arcs if y[i, j].X < 1e-8]

    iteration = 0

    while LB < UB - epsilon:
        iteration += 1
        
        # Inner attacker problem
        
        inner = gp.Model("inner_no_cb")
        inner.Params.OutputFlag = 1 if solver_msg else 0

        x = {
            (i, j): inner.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            for (i, j) in free
        }
        u = {
            (i, j): inner.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"u_{i}_{j}")
            for (i, j) in free
        }
        v = inner.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")

        inner.setObjective(v, GRB.MAXIMIZE)

        inner.addConstr(
            v == gp.quicksum(u[i, j] for (i, j) in free if i == 0),
            name="obj_def",
        )
        inner.addConstr(
            gp.quicksum(
                graph.arcs[i, j].cost_attack * x[i, j] for (i, j) in free
            )
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
                    u[i, j]
                    <= gp.quicksum(u[k, l] for (k, l) in free if k == j),
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

        # Update UB
        if v_star < UB:
            UB = v_star

        inner.dispose()

        
        # Add Benders cut to outer model
        
        outer.addConstr(
            z
            >= v_star
            - gp.quicksum(u_star[i, j] * y[i, j] for (i, j) in free)
            + gp.quicksum(
                u_star[j, k] * w[i, j, k]
                for (i, j) in free
                for (l, k) in free
                if j == l
            ),
            name=f"benders_cut_{iteration}",
        )

        
        # Re-solve outer model
        
        outer.optimize()
        LB = z.X

        free = [(i, j) for (i, j) in arcs if y[i, j].X < 1e-8]

        if iteration > 500:
            break  # Safety cap

    runtime = time.time() - t0
    breach_loss = LB
    interdict = {(i, j): int(round(y[i, j].X)) for (i, j) in arcs}
    outer.dispose()

    return breach_loss, interdict, runtime, iteration
