# ThreatCut

Graph Interdiction & Cybersecurity Optimization Framework

Implementation of the bi-level defender–attacker model from:
> Nandi, Medal, Vadlamani (2016). *Interdicting Attack Graphs to Protect Organizations from Cyber Attacks.* Computers & Operations Research, 75, 118–131.

---

## Requirements

- Python ≥ 3.14
- [uv](https://docs.astral.sh/uv/) (package manager)
- Gurobi licence (the free pip licence bundled with `gurobipy` is sufficient for small instances)

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Lorygold/ThreatCut.git
cd ThreatCut

# 2. Create the virtual environment and install all dependencies
uv sync
```

`uv sync` reads `pyproject.toml`, creates a `.venv` folder, and installs all pinned dependencies (`networkx`, `matplotlib`, `pulp`, `numpy`, `gurobipy`) automatically. No manual `pip install` needed.

---

## Running the project

### Single instance (default parameters)

```bash
uv run python main.py
```

This generates a graph with `L=3, W=3, d=2`, solves it with all methods, prints a comparison table, and shows the CCG convergence plot.

### Custom parameters

```bash
uv run python main.py --L 4 --W 4 --d 3 --B_def 20 --B_att 25 --seed 7
```

### All available flags

| Flag | Default | Description |
|------|---------|-------------|
| `--L` | `3` | Number of levels |
| `--W` | `3` | Nodes per level |
| `--d` | `2` | Out-degree per node |
| `--B_def` | `10` | Defender budget |
| `--B_att` | `15` | Attacker budget |
| `--seed` | `42` | Random seed |
| `--validate` | off | Validate CCG against exact bilevel MIP |
| `--draw` | off | Draw the attack graph before solving |
| `--save_plot PATH` | - | Save convergence plot to file |
| `--no_plot` | off | Skip the convergence plot |
| `--solver_msg` | off | Show Gurobi solver output |

### Reproduce the paper's experiments

```bash
# Quick subset (~2 min)
uv run python experiments/run_experiments.py --quick

# Full experiment (~15–30 min)
uv run python experiments/run_experiments.py

# Save results to CSV
uv run python experiments/run_experiments.py --csv results.csv
```

### Verify the graph generator

```bash
uv run python data/generator.py
```

---

## Project structure

```
ThreatCut/
├── main.py                        # Single-instance entry point
├── pyproject.toml                 # Dependencies and project metadata
├── data/
│   └── generator.py               # Synthetic graph generation
├── model/
│   ├── attack_graph.py            # AttackGraph data structure
│   ├── subproblem.py              # Attacker inner problem (PuLP/CBC)
│   ├── bilevel.py                 # Exact small-instance bilevel MIP
│   ├── algorithm.py               # CCG algorithm + heuristics (PuLP/CBC)
│   ├── new_callbacks.py           # Gurobi solver - lazy constraint callbacks
│   └── new_no_callbacks.py        # Gurobi solver - sequential Benders loop
└── experiments/
    └── run_experiments.py         # Full computational experiment
```
