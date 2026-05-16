# ThreatCut - Attack-Graph Interdiction (Gurobi, free pip licence)

Bi-level defender-attacker optimisation on attack graphs.  
Based on: Nandi, Medal, Vadlamani (2016), *Computers & Operations Research* 75, 118-131.

**Works with the free Gurobi pip licence** — no academic account, no VPN,
no `grbgetkey`. Just `pip install gurobipy`.

---

## Why this instance size?

The free pip licence limits models to **2000 variables and 2000 constraints**.
The critical constraint is the **MinBreachPath master problem** (Model 1):
it accumulates one new variable `u_p` and ~`n_levels` new constraints per
path per iteration. With the paper's 50-node instance this overflows after
just a few iterations.

The chosen instance (**20 nodes, 3 levels, ~44 arcs**) stays well within
limits even at 30 iterations:

| Sub-problem | Variables | Constraints | Limit |
|---|---|---|---|
| Outer MIP (Models 2 & 3) | ~116 | ~72 | 2000  |
| Inner u/x/v (Models 2 & 3) | ~89 | ~111 | 2000  |
| Inner MaxBreachD (Model 1) | ~314 | ~355 | 2000  |
| MinBreachPath master @ 30 iter | ~228 | ~641 | 2000  |

`main.py` verifies these counts at startup and exits cleanly before
calling Gurobi if any limit would be exceeded.

---

## Project structure

```
ThreatCut/
├── main.py                      # Instance generation + benchmark runner
├── requirements.txt
├── README.md
└── model/
    ├── attack_graph.py          # Graph data structures (no solver dependency)
    ├── paper_algorithm.py       # Model 1 - MINMAX exact (paper Section 4)
    ├── model_callbacks.py       # Model 2 - Gurobi lazy-constraint callbacks
    └── model_no_callbacks.py    # Model 3 - Iterative Benders (no callbacks)
```

---

## Installation

```bash
pip install gurobipy      # free pip licence included automatically
```

---

## How to run

```bash
cd threatcut_gurobi
python main.py
```

Expected output:

```
==================================================================
  ThreatCut - Attack-Graph Interdiction Benchmark (Gurobi)
  Nandi, Medal, Vadlamani (2016) - COR 75, 118-131
  Free pip licence  (≤ 2000 vars / 2000 constraints)
==================================================================

[Instance] Building graph (20 nodes, 3 levels, seed=42) ...
  Nodes (incl. virtual source) : 21
  Arcs                         : 44
  Goal nodes                   : 6
  Total reward (UB)            : 5142.3
  Defender budget              : 40.0
  Attacker budget              : 60.0

  Size check  (free-licence limit: 2000, safety threshold: 1900)
  Outer MIP (Models 2 & 3)            :  116 vars /   72 constrs  →  OK 
  Inner u/x/v (Models 2 & 3)          :   89 vars /  111 constrs  →  OK 
  Inner MaxBreachD (Model 1)          :  314 vars /  355 constrs  →  OK 
  MinBreachPath master @30i           :  228 vars /  641 constrs  →  OK 

+----------------------------------------+-------------+-------------+-----------------+------------------+
| Model                                  | Breach Loss | Runtime (s) | Iterations      | Interdicted Arcs |
+----------------------------------------+-------------+-------------+-----------------+------------------+
| Model 1 - Paper MINMAX (MinBreachPath) | 1899.91     | 7.92        | 200             | 0                |
| Model 2 - Gurobi Callbacks             | 1236.47     | 0.90        | N/A (callbacks) | 2                |
| Model 3 - Iterative No-Callbacks       | 1236.47     | 0.13        | 4               | 2                |
+----------------------------------------+-------------+-------------+-----------------+------------------+
```

---

## The three models

| # | File | Description |
|---|------|-------------|
| 1 | `paper_algorithm.py` | **MINMAX exact** (Algorithm 4.3 in the paper). Path-based master MinBreachPath (defender) alternates with MaxBreachD sub-problem (attacker). |
| 2 | `model_callbacks.py` | **Gurobi lazy-constraint callbacks** (port of `new.py`). Single outer MIP; Benders cuts injected via `cbLazy()` at MIPSOL and MIPNODE events. |
| 3 | `model_no_callbacks.py` | **Iterative Benders**. Python loop: solve outer → solve inner → add cut → repeat. |

All three models converge to the same **Breach Loss** value.

---

## Reading the results

| Column | Meaning |
|--------|---------|
| **Breach Loss** | Optimal worst-case attacker reward. All three should match. Lower = better defence. |
| **Runtime (s)** | Wall-clock time in seconds. |
| **Iterations** | Benders outer-loop cycles. Model 2 uses N/A (cuts are internal to Gurobi's B&B). |
| **Interdicted Arcs** | Arcs protected by the defender within budget B_def. |
