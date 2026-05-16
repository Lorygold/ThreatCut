"""
model/model_callbacks.py

Model 2 - Refactored version of new.py.

Single-pass bi-level solver using Gurobi lazy constraints (callbacks).
The outer MIP is solved once; whenever a new integer solution is found
(MIPSOL) or a relaxed node solution exists (MIPNODE), the inner attacker
problem is solved and a Benders-style optimality cut is added on the fly.

Key idea
--------
* Outer variables:
    y[i,j]    - binary, 1 if arc (i,j) is interdicted (defender)
    w[i,j,k]  - binary auxiliary, 1 if both (i,j) and (j,k) are interdicted
    z         - continuous, epigraph variable (worst-case attacker reward)
* At each callback the inner problem is solved over "free" arcs (y=0):
    x[i,j]   - binary, 1 if arc is used
    u[i,j]   - continuous flow value propagated along arc
    v         - total attacker reward
* Cut added:
    z >= v* - sum_{free} u*[i,j]*y[i,j] + sum_{free,free} u*[j,k]*w[i,j,k]
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import gurobipy as gp
from gurobipy import GRB

from model.attack_graph import AttackGraph


def run_callbacks(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], int], float]:
    """
    Solve the MINMAXBREACH problem using Gurobi lazy-constraint callbacks.

    This is a clean, self-contained refactoring of new.py.

    Parameters
    ----------
    graph      : AttackGraph instance
    B_defender : defender budget
    B_attacker : attacker budget
    solver_msg : show Gurobi output for the outer model

    Returns
    -------
    breach_loss  : optimal (or best found) breach loss
    interdict    : interdiction plan {(i,j): 0/1}
    runtime      : wall-clock time in seconds
    """
    arcs = list(graph.arcs.keys())

    # Total reward (upper bound for big-M style flow)
    trew: float = sum(graph.nodes[i].reward for i in graph.nodes)

    
    # Build outer model
    
    outer = gp.Model("outer_callbacks")
    outer.Params.OutputFlag = 1 if solver_msg else 0
    outer.Params.LazyConstraints = 1

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

    # Defender budget constraint
    outer.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_interdict * y[i, j] for i, j in arcs)
        <= B_defender,
        name="budget_defender",
    )

    # Auxiliary variable linking: w[i,j,k] >= y[i,j] + y[j,k] - 1
    for i, j in arcs:
        for l, k in arcs:
            if l == j:
                outer.addConstr(
                    w[i, j, k] >= y[i, j] + y[j, k] - 1,
                    name=f"w_link_{i}_{j}_{k}",
                )

    
    # Inner solver called inside callback
    def _solve_inner(
        free_arcs: list,
    ) -> Tuple[float, dict]:
        """Solve attacker inner problem over free_arcs; return (v*, u* dict)."""
        inner = gp.Model("inner_callback")
        inner.Params.OutputFlag = 0

        x = {
            (i, j): inner.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            for (i, j) in free_arcs
        }
        u = {
            (i, j): inner.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"u_{i}_{j}")
            for (i, j) in free_arcs
        }
        v = inner.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")

        inner.setObjective(v, GRB.MAXIMIZE)

        # v = sum of u[0, j] for arcs leaving source node 0
        inner.addConstr(
            v == gp.quicksum(u[i, j] for (i, j) in free_arcs if i == 0),
            name="obj_def",
        )

        # Attacker budget
        inner.addConstr(
            gp.quicksum(
                graph.arcs[i, j].cost_attack * x[i, j] for (i, j) in free_arcs
            )
            <= B_attacker,
            name="budget_attacker",
        )

        # At most one incoming arc per node (tree structure)
        for node in graph.nodes:
            inner.addConstr(
                gp.quicksum(x[k, l] for (k, l) in free_arcs if l == node) <= 1,
                name=f"in_degree_{node}",
            )

        # Flow propagation constraints
        for i, j in free_arcs:
            inner.addConstr(u[i, j] <= trew * x[i, j], name=f"flow_ub_{i}_{j}")
            if graph.nodes[j].reward <= 0:
                # Intermediate node: flow propagates further
                inner.addConstr(
                    u[i, j]
                    <= gp.quicksum(u[k, l] for (k, l) in free_arcs if k == j),
                    name=f"flow_prop_{i}_{j}",
                )
            else:
                # Goal node: reward collected here
                inner.addConstr(
                    u[i, j] <= graph.nodes[j].reward * x[i, j],
                    name=f"goal_reward_{i}_{j}",
                )

        inner.optimize()

        if inner.Status != GRB.OPTIMAL:
            inner.dispose()
            return 0.0, {}

        v_star = v.X
        u_star = {(i, j): u[i, j].X for (i, j) in free_arcs}
        inner.dispose()
        return v_star, u_star

    
    # Gurobi callback
    
    def callback(model: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIPSOL:
            # Integer solution found
            free = [
                (i, j)
                for (i, j) in arcs
                if model.cbGetSolution(y[i, j]) < 1e-8
            ]
            if not free:
                return
            v_star, u_star = _solve_inner(free)
            # Benders optimality cut
            model.cbLazy(
                z
                >= v_star
                - gp.quicksum(u_star[i, j] * y[i, j] for (i, j) in free)
                + gp.quicksum(
                    u_star[j, k] * w[i, j, k]
                    for (i, j) in free
                    for (l, k) in free
                    if j == l
                )
            )

        elif (
            where == GRB.Callback.MIPNODE
            and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL
        ):
            # LP relaxation at current node is optimal
            free = [
                (i, j)
                for (i, j) in arcs
                if model.cbGetNodeRel(y[i, j]) < 1e-8
            ]
            if not free:
                return
            v_star, u_star = _solve_inner(free)
            model.cbLazy(
                z
                >= v_star
                - gp.quicksum(u_star[i, j] * y[i, j] for (i, j) in free)
                + gp.quicksum(
                    u_star[j, k] * w[i, j, k]
                    for (i, j) in free
                    for (l, k) in free
                    if j == l
                )
            )

    
    # Solve
    
    t0 = time.time()
    outer.optimize(callback)
    runtime = time.time() - t0

    breach_loss = z.X if outer.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else float("inf")
    interdict = {
        (i, j): int(round(y[i, j].X)) for (i, j) in arcs
    }

    outer.dispose()
    return breach_loss, interdict, runtime
