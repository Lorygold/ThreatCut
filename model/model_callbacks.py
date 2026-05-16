"""
model/model_callbacks.py

Model 2 – Benders decomposition via Gurobi lazy-constraint callbacks.

Faithful refactoring of new.py.

A single outer MIP is solved; Gurobi fires a Python callback at every new
integer solution (MIPSOL) and at every optimal LP node relaxation (MIPNODE).
At each event the inner attacker problem is solved and a Benders optimality
cut is injected via model.cbLazy().

Variable counts (n=38 nodes, ~134 arcs):
  Outer MIP: y(134) + w(~424) + z(1) = ~559 vars / ~425 constraints
  Inner MIP: x(<=134) + u(<=134) + v(1) = <=269 vars / <=411 constraints
  Both well within the 2000-var / 2000-constraint free-licence limit.

Outer variables
---------------
  y[i,j]    binary  – 1 if arc (i,j) is interdicted by the defender
  w[i,j,k]  binary  – 1 if both (i,j) and (j,k) are interdicted (auxiliary)
  z         continuous – epigraph variable (worst-case attacker reward)

Inner variables (over free arcs only)
--------------------------------------
  x[i,j]   binary      – 1 if arc used by the attacker
  u[i,j]   continuous  – flow value propagated along the arc
  v         continuous  – total attacker reward
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from model.attack_graph import AttackGraph



# Inner attacker problem

def _solve_inner(
    graph: AttackGraph,
    B_attacker: float,
    trew: float,
    free: List[Tuple[int, int]],
) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """
    Solve the attacker inner problem over free (non-interdicted) arcs.

    Returns
    -------
    v_star : optimal attacker reward
    u_star : {(i,j): u_ij value} – dual flow values used in the Benders cut
    """
    if not free:
        return 0.0, {}

    inner = gp.Model("Inner")
    inner.Params.OutputFlag = 0

    x = {(i, j): inner.addVar(vtype=GRB.BINARY,      name=f"x_{i}_{j}") for (i, j) in free}
    u = {(i, j): inner.addVar(vtype=GRB.CONTINUOUS,   name=f"u_{i}_{j}") for (i, j) in free}
    v = inner.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")

    inner.setObjective(v, GRB.MAXIMIZE)

    # v = sum of u[0,j] for arcs leaving virtual source 0
    inner.addConstr(
        v == gp.quicksum(u[i, j] for (i, j) in free if i == 0),
        name="obj_def",
    )

    # Attacker budget
    inner.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_attack * x[i, j] for (i, j) in free)
        <= B_attacker,
        name="budget_attacker",
    )

    # At most one incoming arc per node (attack is a tree – Lemma 1)
    for node in graph.nodes:
        inner.addConstr(
            gp.quicksum(x[k, l] for (k, l) in free if l == node) <= 1,
            name=f"in_degree_{node}",
        )

    # Flow propagation constraints
    for i, j in free:
        inner.addConstr(u[i, j] <= trew * x[i, j], name=f"flow_ub_{i}_{j}")
        if graph.nodes[j].reward <= 0:
            # Intermediate node: reward propagates further down the path
            inner.addConstr(
                u[i, j] <= gp.quicksum(u[k, l] for (k, l) in free if k == j),
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
    u_star = {(i, j): u[i, j].X for (i, j) in free}
    inner.dispose()
    return v_star, u_star



# Entry point

def run_callbacks(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], int], float]:
    """
    Solve MINMAXBREACH using Gurobi lazy-constraint callbacks.

    The outer MIP is solved once; Benders cuts are injected via cbLazy()
    at every MIPSOL event (integer solution found) and at every MIPNODE
    event where the LP relaxation is optimal.

    Returns
    -------
    breach_loss : optimal breach loss
    interdict   : interdiction plan {(i,j): 0/1}
    runtime     : wall-clock seconds
    """
    arcs = list(graph.arcs.keys())
    trew: float = sum(graph.nodes[i].reward for i in graph.nodes)

    
    # Build outer MIP
    
    outer = gp.Model("Outer_CB")
    outer.Params.OutputFlag = 1 if solver_msg else 0
    outer.Params.LazyConstraints = 1

    y = {(i, j): outer.addVar(vtype=GRB.BINARY,     name=f"y_{i}_{j}") for (i, j) in arcs}
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

    # Auxiliary linking: w[i,j,k] >= y[i,j] + y[j,k] - 1
    for i, j in arcs:
        for l, k in arcs:
            if l == j:
                outer.addConstr(
                    w[i, j, k] >= y[i, j] + y[j, k] - 1,
                    name=f"w_link_{i}_{j}_{k}",
                )

    
    # Gurobi callback
    
    def _callback(model: gp.Model, where: int) -> None:
        """Inject Benders optimality cuts at integer solutions and LP nodes."""

        if where == GRB.Callback.MIPSOL:
            free = [
                (i, j) for (i, j) in arcs
                if model.cbGetSolution(y[i, j]) < 1e-8
            ]
        elif (
            where == GRB.Callback.MIPNODE
            and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL
        ):
            free = [
                (i, j) for (i, j) in arcs
                if model.cbGetNodeRel(y[i, j]) < 1e-8
            ]
        else:
            return

        if not free:
            return

        v_star, u_star = _solve_inner(graph, B_attacker, trew, free)

        # Benders cut:  z >= v* - sum u*[i,j]*y[i,j] + sum u*[j,k]*w[i,j,k]
        model.cbLazy(
            z >= v_star
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
    outer.optimize(_callback)
    runtime = time.time() - t0

    breach_loss = z.X if outer.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else float("inf")
    interdict = {(i, j): int(round(y[i, j].X)) for (i, j) in arcs}
    outer.dispose()
    return breach_loss, interdict, runtime
