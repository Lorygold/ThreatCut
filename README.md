# Attack-Graph Interdiction - Bi-Level Optimization

Implementation of the defender-attacker bi-level network interdiction model from:

> Nandi, Medal, Vadlamani (2016). *Interdicting attack graphs to protect organizations from cyber attacks: A bi-level defender–attacker model.* Computers & Operations Research, 75, 118–131.

---

## Project structure

```
attack_graph_project/
├── main.py                        # Instance generation + benchmark runner
├── README.md
└── model/
    ├── attack_graph.py            # Graph data structures
    ├── paper_algorithm.py         # Model 1
    ├── model_callbacks.py         # Model 2
    └── model_no_callbacks.py      # Model 3
```

---

## The three models

| # | File | Description |
|---|------|-------------|
| 1 | `paper_algorithm.py` | **MINMAX exact algorithm** from the paper. Alternates between a path-based master problem (MINBREACHPATH) and an attacker sub-problem (MaxBreachD) until the optimality gap closes. |
| 2 | `model_callbacks.py` | **Gurobi lazy-constraint callbacks** (refactoring of `new.py`). One single outer MIP solve; Benders cuts are injected at every integer solution (MIPSOL) and every LP node (MIPNODE) via Gurobi's callback API. |
| 3 | `model_no_callbacks.py` | **Iterative Benders without callbacks** (refactoring of `new_Gurobi_no_callbacks.py`). The outer and inner MIPs alternate in a plain Python loop; a Benders cut is added to the outer model after each inner solve, which is then re-optimised. |

---

## Requirements

```
gurobipy   # Gurobi solver (licence required)
python >= 3.9
```

---

## How to run

```bash
cd attack_graph_project
python main.py
```

The script builds a 50-node, 5-level synthetic attack graph (seed = 42, matching the paper's low-level parameter settings) with B\_defender = 75 and B\_attacker = 125, runs all three models, and prints the comparison table.

---

## Reading the results

| Column | Meaning |
|--------|---------|
| **Breach Loss** | Optimal worst-case attacker reward. All three models should converge to the same value. Lower means better defence. |
| **Runtime (s)** | Total wall-clock time. Model 2 is typically fastest on small instances; Model 1 may require more iterations. |
| **Iterations** | Number of outer-loop cycles (master + sub-problem solves). Model 2 reports "N/A" because it uses a single outer solve with callbacks. |
| **Interdicted Arcs** | Number of arcs the defender chooses to protect within the budget. |

**Expected outcome:** all three models produce the same (or very close) Breach Loss, confirming correctness. Runtime differences reflect the trade-off between the overhead of Gurobi callbacks (Model 2) versus explicit Python looping (Models 1 and 3).
