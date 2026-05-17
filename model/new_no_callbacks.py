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

Implementation suggested by Roberto Montemanni
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

    Returns
    -------
    (inner_model, x_vars, u_vars, v_var)
    """
    global L,W # RM
    inner = gp.Model("inner")
    inner.Params.OutputFlag = 0  # suppress solver output

    # x[i,j] = 1 if the attacker uses arc (i,j) in the attack plan
    x = {(i, j): inner.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
         for (i, j) in free_arcs}

    # u[i,j] = profit carried along arc (i,j) in the optimal attack plan
    u = {(i, j): inner.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"u_{i}_{j}")
         for (i, j) in free_arcs}

    # v = total profit collected at the root (the objective)
    v = inner.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")

    # Objective: maximise profit collected at root node (node id = 0)
    inner.setObjective(v, GRB.MAXIMIZE)

    # --- Constraints ---

    # Total profit at root = sum of profits on arcs leaving the root node
    inner.addConstr(
        v == gp.quicksum(u[i, j] for (i, j) in free_arcs if i == 0),
        name="root_profit"
    )

    # Attacker budget: sum of attack costs on used arcs must not exceed B_a
    inner.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_attack * x[i, j]
                    for (i, j) in free_arcs) <= B_attacker,
        name="attacker_budget"
    )

    # Arborescence constraint: at most one arc can enter each non-root node.
    # This forces the attack plan to be a tree (arborescence) rooted at node 0,
    # which generalises the single-path model of Nandi et al. (2016) by allowing
    # the attacker to simultaneously pursue multiple goal nodes.
    for node_id in graph.nodes:
        in_arcs = [(i, j) for (i, j) in free_arcs if j == node_id]
        if in_arcs:
            inner.addConstr(
                gp.quicksum(x[i, j] for (i, j) in in_arcs) <= 1,
                name=f"arborescence_{node_id}"
            )

    # Profit propagation constraints
    for (i, j) in free_arcs:
        # u[i,j] is zero unless arc (i,j) is active (big-M upper bound)
        inner.addConstr(u[i, j] <= M * x[i, j], name=f"bigM_{i}_{j}")

        if graph.nodes[j].reward > 0:
            # j is a goal (leaf) node: profit is capped at the node reward
            inner.addConstr(
                u[i, j] <= graph.nodes[j].reward * x[i, j],
                name=f"goal_reward_{i}_{j}"
            )
        else:
            # j is an intermediate node: profit on arc (i,j) cannot exceed
            # the total profit collected by arcs leaving j (forward flow)
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

    return inner, x, u, v


def run_new_no_callbacks(
    graph: AttackGraph,
    B_defender: float,
    B_attacker: float,
    Lp: int, # RM qui va gestito meglio...
    Wp: int, # RM qui va gestito meglio...
    solver_msg: bool = False,
) -> Tuple[float, Dict[Tuple[int, int], int], int]:
    """
    Solve the bilevel cyber attack interdiction problem using an explicit
    sequential Benders decomposition loop (no Gurobi callbacks).

    Algorithm
    ---------
    Iteration 0:
        Solve the outer problem with no Benders cuts → LB = z*, extract plan y*
        Set UB = sum of all goal rewards (trivial upper bound)

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
    graph       : AttackGraph
    B_defender  : float  Defender's interdiction budget.
    B_attacker  : float  Attacker's traversal budget.
    solver_msg  : bool   Show Gurobi output if True.

    Returns
    -------
    (breach_loss, x_optimal, n_iterations)
        breach_loss  : optimal worst-case attacker profit
        x_optimal    : dict {(i,j): 0 or 1} - optimal interdiction plan
        n_iterations : number of Benders iterations performed
    """
    global L,W # RM
    L=Lp; W=Wp # RM
    
    arcs = list(graph.arcs.keys())

    # big-M = sum of all goal node rewards (maximum possible attacker profit)
    M = sum(n.reward for n in graph.nodes.values())

    # Build the outer (defender) problem
    outer = gp.Model("outer_no_callbacks")
    outer.Params.OutputFlag = 1 if solver_msg else 0

    # y[i,j] = 1 if the defender interdicts arc (i,j)
    y = {(i, j): outer.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
         for (i, j) in arcs}

    # w[i,j,k] = 1 if BOTH arcs (i,j) AND (j,k) are interdicted.
    # Correction variable to avoid double-counting in the Benders cut:
    # when two consecutive arcs are interdicted, the cut term
    # -u_{ij}*y_{ij} already removes arc (i,j)'s contribution, but also
    # incorrectly removes it from arc (j,k)'s downstream profit.
    # The +u_{jk}*w_{ijk} term compensates for this.
    w = {
        (i, j, k): outer.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}_{k}")
        for (i, j) in arcs
        for (l, k) in arcs if j == l
    }

    # z = worst-case attacker profit (variable we minimise)
    z = outer.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    # Objective: minimise worst-case attacker profit
    outer.setObjective(z, GRB.MINIMIZE)

    # Defender budget constraint
    outer.addConstr(
        gp.quicksum(graph.arcs[i, j].cost_interdict * y[i, j]
                    for (i, j) in arcs) <= B_defender,
        name="defender_budget"
    )

    # w alignment constraints: w[i,j,k] = 1 when both y[i,j]=1 and y[j,k]=1
    
    # RM Qui aggiunti 2 vincoli

    for (i, j) in arcs:
        for (l, k) in arcs:
            if l == j:
                outer.addConstr(
                    w[i, j, k] >= y[i, j] + y[j, k] - 1,
                    name=f"w_align_{i}_{j}_{k}"
                )
                outer.addConstr( # RM
                    w[i, j, k] <= y[i, j],
                    name=f"w_align_{i}_{j}"
                )
                outer.addConstr( # RM
                    w[i, j, k] <= y[j, k],
                    name=f"w_align_{j}_{k}"
                )

    # Initial solve (no Benders cuts yet - z is unconstrained from above)
    outer.optimize()

    lb = z.X
    ub = M  # trivial upper bound: attacker can collect at most sum of all rewards

    # Free arcs: arcs NOT interdicted in the current outer solution
    free = [(i, j) for (i, j) in arcs if y[i, j].X < 1e-7]

    print(f"  Init  |  LB = {lb:.4f}  |  UB = {ub:.4f}")

    n_iter = 0

    # Benders loop
    while lb < ub - 1e-7:
        n_iter += 1

        # --- Solve inner (attacker) problem ---
        inner, x_vars, u_vars, v_var = _build_inner_problem(
            graph, free, B_attacker, M
        )
        inner.optimize()

        attacker_profit = v_var.X

        # Update upper bound: the attacker's best response to the current
        # outer plan is a valid upper bound on the optimal breach loss.
        if attacker_profit < ub:
            ub = attacker_profit

        # --- Add Benders cut to the outer problem ---
        # The cut says: no matter what interdiction plan y the defender chooses,
        # z must be at least the attacker's profit under that plan.
        # The u* values from the inner solution quantify how much each
        # interdicted arc reduces the attacker's profit.
        outer.addConstr(
            z >= attacker_profit
            # Each interdicted arc (i,j) reduces attacker profit by u*_{ij}
            - gp.quicksum(u_vars[i, j].X * y[i, j] for (i, j) in free)
            # Correction for consecutive interdictions (avoids double-counting)
            + gp.quicksum(
                u_vars[j, k].X * w[i, j, k]
                for (i, j) in free
                for (l, k) in free if j == l
            ),
            name=f"benders_cut_{n_iter}"
        )

        inner.dispose()

        # --- Re-solve outer problem with the new cut ---
        outer.optimize()

        lb = z.X

        # Extract new free arc set for next iteration
        free = [(i, j) for (i, j) in arcs if y[i, j].X < 1e-7]

        print(f"  Iter {n_iter:3d}  |  LB = {lb:.4f}  |  UB = {ub:.4f}")

    print(f"\nConverged after {n_iter} iteration(s).  Breach loss = {ub:.4f}")

    # Extract optimal interdiction plan
    x_optimal = {(i, j): int(round(y[i, j].X or 0)) for (i, j) in arcs}

    outer.dispose()

    return ub, x_optimal, n_iter
