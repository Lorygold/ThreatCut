"""
model/new_callbacks.py

Gurobi-based implementation of the bilevel defender-attacker interdiction
model using LAZY CONSTRAINTS via Gurobi callbacks.

This is the most efficient implementation: instead of solving the outer
(defender) and inner (attacker) problems in a sequential loop, the inner
problem is solved INSIDE Gurobi's branch-and-bound tree via callbacks.

Two callback events are exploited:
  - GRB.Callback.MIPSOL  : triggered when Gurobi finds a new integer-feasible
                            solution for the outer problem. The inner problem
                            is solved exactly here, and a Benders cut is added
                            as a lazy constraint to tighten the outer model.
  - GRB.Callback.MIPNODE : triggered at each B&B node when the LP relaxation
                            is optimal. The inner problem is solved on the LP
                            relaxed solution (fractional y values are rounded
                            down to 0 to build the free arc set F). The cut
                            added here is valid but not necessarily tight -
                            it strengthens the LP bound and prunes more nodes.

The model introduces the auxiliary variable w_{ijk} to handle the correction
factor when two consecutive arcs (i,j) and (j,k) are both interdicted,
avoiding double-counting of the interdiction effect in the Benders cut.

Implementation suggested by Roberto Montemanni.
"""

from __future__ import annotations

from typing import Dict, Tuple

import gurobipy as gp
from gurobipy import GRB

from model.attack_graph import AttackGraph



# Module-level shared state for the outer callback.
# Gurobi callbacks cannot receive extra arguments - storing state at module
# level is the standard pattern for Gurobi callbacks in Python.

_prob:      gp.Model        # outer (defender) Gurobi model
_y:         Dict            # outer binary interdiction variables y[i,j]
_w:         Dict            # outer auxiliary variables w[i,j,k]
_z:         gp.Var          # outer continuous variable z (worst-case attacker profit)
_arcs:      list            # list of arc tuples (i, j)
_graph:     AttackGraph     # attack graph (read-only inside callback)
_B_attacker: float          # attacker budget (read-only inside callback)
_M:         float           # big-M = sum of all goal node rewards
_L:         int             # number of levels (for valid inequality)
_W:         int             # nodes per level (for valid inequality)

# Inner-problem variables - set by _solve_inner_and_add_cut, read by _inner_callback.
_u:         Dict            # inner u[i,j] variables
_v:         gp.Var          # inner v variable
_free_arcs: list            # free arcs for the current inner solve



# Inner callback: fires when the inner MIP finds a new integer solution


def _inner_callback(model: gp.Model, where: int) -> None:
    """
    Gurobi callback for the INNER (attacker) MIP.

    At every integer solution of the inner problem (MIPSOL), immediately
    add the corresponding Benders cut to the OUTER model via cbLazy().

    This is the key design: cuts are generated at every integer feasible
    solution of the inner MIP, not only at its optimum, tightening the
    outer model much faster.
    """
    global _prob, _y, _w, _z, _u, _v, _free_arcs

    if where == GRB.Callback.MIPSOL:
        _prob.cbLazy(
            _z >= model.cbGetSolution(_v)
            - gp.quicksum(
                model.cbGetSolution(_u[i, j]) * _y[i, j]
                for (i, j) in _free_arcs
            )
            + gp.quicksum(
                model.cbGetSolution(_u[j, k]) * _w[i, j, k]
                for (i, j) in _free_arcs
                for (l, k) in _free_arcs if j == l
            )
        )



# Solve inner and inject Benders cut (called from the outer callback)

def _solve_inner_and_add_cut(free_arcs: list) -> None:
    """
    Build and solve the attacker inner MIP for the given free-arc set, then
    add the Benders cut to the outer model via cbLazy().

    The inner MIP is solved with its own callback (_inner_callback), so cuts
    are generated at every integer feasible solution of the inner problem.

    The Benders cut has the form:

        z >= v* - sum_{(i,j) in F} u*_{ij} * y_{ij}
                + sum_{(i,j),(j,k) in F} u*_{jk} * w_{ijk}

    where F is the set of arcs not interdicted by the current outer solution,
    v* is the inner objective value, and u* are the arc profit variables.

    The w correction term compensates for double-counting: when two consecutive
    arcs (i,j) and (j,k) are both interdicted, the term -u*_{ij}*y_{ij} already
    removes (i,j)'s contribution, but also incorrectly removes the downstream
    profit flowing through j. The +u*_{jk}*w_{ijk} term adds it back.

    Parameters
    ----------
    free_arcs : list of (i, j) tuples - arcs where y_{ij} = 0
    """
    global _u, _v, _free_arcs
    _free_arcs = free_arcs

    inner = gp.Model("inner")
    inner.Params.OutputFlag = 0

    x = {(i, j): inner.addVar(vtype=GRB.BINARY,     name=f"x_{i}_{j}") for (i, j) in free_arcs}
    _u = {(i, j): inner.addVar(vtype=GRB.CONTINUOUS, name=f"u_{i}_{j}") for (i, j) in free_arcs}
    _v = inner.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")

    inner.setObjective(_v, GRB.MAXIMIZE)

    # Total profit at root = sum of u on arcs leaving node 0
    inner.addConstr(
        _v == gp.quicksum(_u[i, j] for (i, j) in free_arcs if i == 0),
        name="root_profit",
    )

    # Attacker budget
    inner.addConstr(
        gp.quicksum(_graph.arcs[i, j].cost_attack * x[i, j] for (i, j) in free_arcs)
        <= _B_attacker,
        name="attacker_budget",
    )

    # Arborescence: at most one incoming arc per node
    for node_id in _graph.nodes:
        in_arcs = [(i, j) for (i, j) in free_arcs if j == node_id]
        if in_arcs:
            inner.addConstr(
                gp.quicksum(x[i, j] for (i, j) in in_arcs) <= 1,
                name=f"arborescence_{node_id}",
            )

    # Profit propagation
    for (i, j) in free_arcs:
        inner.addConstr(_u[i, j] <= _M * x[i, j], name=f"bigM_{i}_{j}")
        if _graph.nodes[j].reward > 0:
            inner.addConstr(
                _u[i, j] <= _graph.nodes[j].reward * x[i, j],
                name=f"goal_reward_{i}_{j}",
            )
        else:
            out_arcs = [(l, k) for (l, k) in free_arcs if l == j]
            if out_arcs:
                inner.addConstr(
                    _u[i, j] <= gp.quicksum(_u[j, k] for (j, k) in out_arcs),
                    name=f"flow_fwd_{i}_{j}",
                )

    # Valid inequality: total flow crossing each inter-level cut equals v.
    # Applied once per level, not once per arc.
    # Only meaningful for regular L×W graphs (skip when _L or _W is 0).
    if _L > 0 and _W > 0:
        for l in range(_L):
            idx  = 1 + (l - 1) * _W
            idx2 = 1 + l * _W
            inner.addConstr(
                gp.quicksum(
                    _u[i, j]
                    for i in range(idx,  idx  + _W)
                    for j in range(idx2, idx2 + _W)
                    if (i, j) in x
                ) == _v,
                name=f"level_flow_{l}",
            )

    # Solve inner with its own callback so cuts fire at every integer solution
    inner.optimize(_inner_callback)
    inner.dispose()



# Outer callback

def _callback(model: gp.Model, where: int) -> None:
    """
    Outer Gurobi callback.

    MIPSOL  - new integer-feasible outer solution: solve inner on exact y values.
    MIPNODE - optimal LP relaxation at B&B node: solve inner on fractional y values
              (arcs with value < 0.1 treated as non-interdicted).
    """
    if where == GRB.Callback.MIPSOL:
        free = [
            (i, j) for (i, j) in _arcs
            if model.cbGetSolution(_y[i, j]) < 0.1
        ]
        _solve_inner_and_add_cut(free)

    elif (
        where == GRB.Callback.MIPNODE
        and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL
    ):
        free = [
            (i, j) for (i, j) in _arcs
            if model.cbGetNodeRel(_y[i, j]) < 0.1
        ]
        _solve_inner_and_add_cut(free)



# Public entry point

def run_new_callbacks(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    Lp: int,
    Wp: int,
    solver_msg: bool = False,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Solve the bilevel cyber attack interdiction problem using Gurobi with
    lazy Benders cuts added via callbacks.

    The outer (defender) problem is solved by Gurobi's B&B engine.
    At each integer solution and at each LP node, the inner (attacker)
    problem is solved and a Benders cut is injected directly into the
    B&B tree as a lazy constraint.

    This is more efficient than the sequential loop approach because:
      - Cuts are added throughout the B&B tree, not just at the root.
      - Gurobi can exploit the tighter bounds to prune more branches.
      - No re-solving from scratch at each iteration.

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
    (breach_loss, runtime_seconds)
        breach_loss : optimal worst-case attacker profit after interdiction
        runtime_s   : Gurobi wall-clock solve time in seconds
    """
    global _prob, _y, _w, _z, _arcs, _graph, _B_attacker, _M, _L, _W

    _graph      = graph
    _B_attacker = B_attacker
    _arcs       = list(graph.arcs.keys())
    _M          = sum(n.reward for n in graph.nodes.values())
    _L          = Lp
    _W          = Wp

    _prob = gp.Model("outer_callback")
    _prob.Params.OutputFlag    = 1 if solver_msg else 0
    _prob.Params.LazyConstraints = 1   # required for cbLazy

    _y = {(i, j): _prob.addVar(vtype=GRB.BINARY,     name=f"y_{i}_{j}") for (i, j) in _arcs}
    _w = {
        (i, j, k): _prob.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}_{k}")
        for (i, j) in _arcs
        for (l, k) in _arcs if j == l
    }
    _z = _prob.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    _prob.setObjective(_z, GRB.MINIMIZE)

    _prob.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_interdict * _y[i, j] for (i, j) in _arcs)
        <= B_defender,
        name="defender_budget",
    )

    for (i, j) in _arcs:
        for (l, k) in _arcs:
            if l == j:
                _prob.addConstr(_w[i, j, k] >= _y[i, j] + _y[j, k] - 1, name=f"w_align_{i}_{j}_{k}")
                _prob.addConstr(_w[i, j, k] <= _y[i, j],                  name=f"w_align_ij_{i}_{j}_{k}")
                _prob.addConstr(_w[i, j, k] <= _y[j, k],                  name=f"w_align_jk_{i}_{j}_{k}")

    _prob.optimize(_callback)

    breach_loss = _z.X
    runtime_s   = _prob.Runtime

    if verbose:
        print(f"Optimal breach loss: {breach_loss:.4f}  |  Runtime: {runtime_s:.3f}s")

    _prob.dispose()
    return breach_loss, runtime_s
