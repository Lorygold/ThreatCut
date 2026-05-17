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

which introduces the auxiliary variable w_{ijk} to handle the correction
factor when two consecutive arcs (i,j) and (j,k) are both interdicted,
avoiding the double-counting of the interdiction effect in the Benders cut.

Implementation suggested by Roberto Montemanni
"""

from __future__ import annotations

from typing import Dict, Tuple

import gurobipy as gp
from gurobipy import GRB

from model.attack_graph import AttackGraph


# Module-level globals used inside the callback.
# Gurobi callbacks cannot receive extra arguments, so shared state is stored
# at module level.  This is the standard pattern for Gurobi callbacks in Python.

_prob: gp.Model          # outer (defender) Gurobi model
_y: Dict                 # outer binary interdiction variables y[i,j]
_w: Dict                 # outer auxiliary variables w[i,j,k]
_z: gp.Var              # outer continuous variable z (worst-case attacker profit)
_arcs: list              # list of arc tuples (i, j)
_graph: AttackGraph      # attack graph (read-only inside callback)
_B_attacker: float       # attacker budget (read-only inside callback)
_M: float                # big-M = sum of all goal node rewards

def inner_callback(model: gp.Model, where: int) -> None: # RM
    global _prob, _y, _w, _z
    
    global u,v,free_arcs2 # RM
    if where == GRB.Callback.MIPSOL:
    # --- Add Benders cut as lazy constraint ---
    # The cut tightens z from below: the outer z must be at least as large
    # as the attacker's profit under any valid interdiction plan.
        _prob.cbLazy(
            _z >= model.cbGetSolution(v)
            # Subtract profit lost due to interdicted arcs
            - gp.quicksum(model.cbGetSolution(u[i, j]) * _y[i, j] for (i, j) in free_arcs2)
            # Correction: add back the over-counted reduction for consecutive interdictions
            + gp.quicksum(
                model.cbGetSolution(u[j, k]) * _w[i, j, k]
                for (i, j) in free_arcs2
                for (l, k) in free_arcs2 if j == l
            )
        )

def _solve_inner_and_add_cut(free_arcs: list) -> None:
    """
    Solve the attacker's inner problem on the set of non-interdicted arcs
    and add the resulting Benders cut as a lazy constraint to the outer model.

    This function is called from within the Gurobi callback, so it uses
    _prob.cbLazy() to add the cut rather than _prob.addConstr().

    The Benders cut has the form (equation 4 in the course notes):

        z >= v^p
             - sum_{(i,j) in F} u^p_{ij} * y_{ij}
             + sum_{(i,j),(j,k) in F} u^p_{jk} * w_{ijk}

    where:
        v^p   = optimal inner objective (attacker profit with no interdiction)
        u^p   = optimal profit values on each arc in the inner solution
        F     = set of arcs NOT interdicted by the current outer solution

    The correction term with w_{ijk} compensates for the fact that when two
    consecutive arcs (i,j) and (j,k) are BOTH interdicted, the term
    u^p_{ij} * y_{ij} would remove the profit on arc (i,j), but that profit
    was already zero because (j,k) is also cut - so we would subtract too much.
    The w term adds back this over-counted reduction.

    Parameters
    ----------
    free_arcs : list of (i, j) tuples
        Arcs that are NOT interdicted in the current outer solution
        (i.e., arcs where y_{ij} = 0).
    """
    global u,v,free_arcs2 # RM
    free_arcs2=free_arcs # RM

    # Build the inner (attacker) problem
    inner = gp.Model("inner")
    inner.Params.OutputFlag = 0  # silence inner solver output
    inner.Params.LazyConstraints = 1 # RM
    
    # --- Inner decision variables ---
    # x[i,j] = 1 if the attacker uses arc (i,j) in the attack plan
    x = {(i, j): inner.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
         for (i, j) in free_arcs}

    # u[i,j] = profit carried along arc (i,j) in the attack plan
    # This flows rewards from goal nodes back towards the root, following
    # the active arborescence structure.
    u = {(i, j): inner.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"u_{i}_{j}")
         for (i, j) in free_arcs}

    # v = total profit collected at the root (objective variable)
    v = inner.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")

    # --- Objective: maximise total profit collected at root node (id=0) ---
    inner.setObjective(v, GRB.MAXIMIZE)

    # --- Constraints ---

    # Total profit collected at root = sum of profits on arcs leaving root
    inner.addConstr(
        v == gp.quicksum(u[i, j] for (i, j) in free_arcs if i == 0),
        name="root_profit"
    )

    # Attacker budget: total cost of arcs used must not exceed B_a
    inner.addConstr(
        gp.quicksum(_graph.arcs[i, j].cost_attack * x[i, j]
                    for (i, j) in free_arcs) <= _B_attacker,
        name="attacker_budget"
    )

    # Arborescence constraint: at most one arc can enter each non-root node.
    # This ensures the attack plan forms a tree rooted at node 0.
    for node_id in _graph.nodes:
        in_arcs = [(i, j) for (i, j) in free_arcs if j == node_id]
        if in_arcs:
            inner.addConstr(
                gp.quicksum(x[i, j] for (i, j) in in_arcs) <= 1,
                name=f"arborescence_{node_id}"
            )

    # Profit propagation along the arborescence
    for (i, j) in free_arcs:
        # u[i,j] can be non-zero only if arc (i,j) is active (big-M bound)
        inner.addConstr(u[i, j] <= _M * x[i, j], name=f"bigM_{i}_{j}")

        if _graph.nodes[j].reward > 0:
            # Node j is a goal node: profit on arc entering j is capped at r_j
            inner.addConstr(
                u[i, j] <= _graph.nodes[j].reward * x[i, j],
                name=f"goal_reward_{i}_{j}"
            )
        else:
            # Intermediate node: profit flows forward from j to its successors
            out_arcs = [(l, k) for (l, k) in free_arcs if l==j] # RM Attenzione qui: vanno presi solo gli archi uscenti da j!
            if out_arcs:
                inner.addConstr(
                    u[i, j] <= gp.quicksum(u[j, k] for (j, k) in out_arcs),
                    name=f"flow_fwd_{i}_{j}"
                )
        for l in range(L): # RM this is a valid inequality to speed up computation. Is it really necessary to use W or we have a "nodes in a certain level" concept?
            idx=1+(l-1)*W
            idx2=1+(l)*W
            inner.addConstr(gp.quicksum(u[i,j] for i in range(idx,idx+W)for j in range(idx2,idx2+W) if (i,j) in x)==v)
            
    inner.optimize(inner_callback) # RM

    inner.dispose()


def _callback(model: gp.Model, where: int) -> None:
    """
    Gurobi callback function.

    Called by Gurobi at various points during the B&B solve.
    We intercept two events:

    MIPSOL  - a new integer-feasible outer solution has been found.
              The inner problem is solved on the exact interdiction plan,
              and a tight Benders cut is added.

    MIPNODE - the LP relaxation at a B&B node has been solved to optimality.
              The inner problem is solved on the fractionally-relaxed plan
              (arcs with LP value < epsilon are treated as non-interdicted),
              and a valid (but possibly not tight) cut is added to strengthen
              the LP relaxation bound.
    """
    if where == GRB.Callback.MIPSOL:
        # Extract arcs NOT interdicted in the current integer solution
        free = [
            (i, j) for (i, j) in _arcs
            if _prob.cbGetSolution(_y[i, j]) < 1e-1 # RM era rischioso a 1e-7 (my fault)
        ]
        _solve_inner_and_add_cut(free)

    elif (where == GRB.Callback.MIPNODE and
          _prob.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
        # Extract arcs NOT interdicted in the LP relaxation at this B&B node
        # (fractional values close to 0 are treated as "not interdicted")
        free = [
            (i, j) for (i, j) in _arcs
            if _prob.cbGetNodeRel(_y[i, j]) < 1e-1 # RM era rischioso a 1e-7 (my fault)
        ]
        _solve_inner_and_add_cut(free)


def run_new_callbacks(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    Lp: int, # RM qui va gestito meglio...
    Wp: int, # RM qui va gestito meglio...
    solver_msg: bool = False,
) -> Tuple[float, float]:
    """
    Solve the bilevel cyber attack interdiction problem using Gurobi with
    lazy Benders cuts added via callbacks.

    The outer (defender) problem is solved by Gurobi's B&B engine.
    At each integer solution and at each LP node, the inner (attacker)
    problem is solved and a Benders cut is injected directly into the
    B&B tree as a lazy constraint.

    This is more efficient than the sequential loop approach because:
    - Cuts are added throughout the B&B tree, not just at the root
    - Gurobi can exploit the tighter bounds to prune more branches
    - No re-solving from scratch at each iteration

    Parameters
    ----------
    graph       : AttackGraph
    B_defender  : float  Defender's interdiction budget.
    B_attacker  : float  Attacker's traversal budget.
    solver_msg  : bool   Show Gurobi output if True.

    Returns
    -------
    (breach_loss, runtime_seconds)
        breach_loss : optimal worst-case attacker profit after interdiction
        runtime_s   : Gurobi wall-clock solve time in seconds
    """
    global _prob, _y, _w, _z, _arcs, _graph, _B_attacker, _M,L,W # RM
    L=Lp; W=Wp # RM
    # Store shared state for the callback
    _graph = graph
    _B_attacker = B_attacker
    _arcs = list(graph.arcs.keys())

    # big-M = sum of all goal node rewards (maximum possible attacker profit)
    _M = sum(n.reward for n in graph.nodes.values())

    # Build the outer (defender) problem

    _prob = gp.Model("outer_callback")
    _prob.Params.OutputFlag = 1# if solver_msg else 0
    # REQUIRED: lazy constraints can only be added via callbacks when this
    # parameter is enabled.
    _prob.Params.LazyConstraints = 1

    # --- Outer decision variables ---
    # y[i,j] = 1 if the defender interdicts arc (i,j)
    _y = {(i, j): _prob.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
          for (i, j) in _arcs}

    # w[i,j,k] = 1 if BOTH arcs (i,j) AND (j,k) are interdicted.
    # Used as a correction factor in the Benders cut to avoid
    # double-counting when two consecutive arcs are blocked.
    _w = {
        (i, j, k): _prob.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}_{k}")
        for (i, j) in _arcs
        for (l, k) in _arcs if j == l
    }

    # z = worst-case attacker profit (the variable we minimise)
    _z = _prob.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    # --- Objective ---
    _prob.setObjective(_z, GRB.MINIMIZE)

    # --- Constraints ---

    # Defender budget
    _prob.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_interdict * _y[i, j]
                    for (i, j) in _arcs) <= B_defender,
        name="defender_budget"
    )

    # w alignment: w[i,j,k] = 1 whenever BOTH y[i,j]=1 AND y[j,k]=1.
    # Equivalently: w[i,j,k] >= y[i,j] + y[j,k] - 1
    
    # RM Qui aggiunti 2 vincoli
    
    for (i, j) in _arcs:
        for (l, k) in _arcs:
            if l == j:
                _prob.addConstr(
                    _w[i, j, k] >= _y[i, j] + _y[j, k] - 1,
                    name=f"w_align_{i}_{j}_{k}"
                )
                _prob.addConstr( # RM
                    _w[i, j, k] <= _y[i, j],
                    name=f"w_align_{i}_{j}"
                )
                _prob.addConstr( # RM
                    _w[i, j, k] <= _y[j, k],
                    name=f"w_align_{j}_{k}"
                )

    # Solve with callback
    _prob.optimize(_callback)

    breach_loss = _z.X
    runtime_s = _prob.Runtime

    print(f"Optimal breach loss: {breach_loss:.4f}  |  Runtime: {runtime_s:.3f}s")

    _prob.dispose()
    return breach_loss, runtime_s
