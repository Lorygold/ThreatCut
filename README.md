# ThreatCut

Graph Interdiction & Cybersecurity Optimization Framework

Upgraded implementation of the bi-level defender–attacker model from:
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

`uv sync` reads `pyproject.toml`, creates a `.venv` folder, and installs all pinned dependencies (`networkx`, `matplotlib`, `pulp`, `numpy`, `gurobipy`) automatically.

---

## Running the project

### Single instance (paper defaults)

```bash
uv run python main.py
```

Generates a graph with `L=3, W=3, d=18` using the paper's **low** cost preset, solves with all methods, prints a comparison table, and shows the CCG convergence plot.

### Custom parameters

```bash
uv run python main.py --L 5 --W 10 --d 100 --B_def 75 --B_att 125
```

### Use the paper's "high" cost preset

```bash
uv run python main.py --L 5 --W 10 --d 100 --B_def 75 --B_att 125 --high_costs
```

### All available flags

| Flag | Default | Description |
|------|---------|-------------|
| `--L` | `3` | Number of levels |
| `--W` | `3` | Nodes per level |
| `--d` | `18` | Target non-source arc count |
| `--B_def` | `30` | Defender budget |
| `--B_att` | `50` | Attacker budget |
| `--seed` | `42` | Random seed |
| `--mina` | `10` | Min attack cost |
| `--maxa` | `30` | Max attack cost |
| `--mind` | `10` | Min interdict cost |
| `--maxd` | `30` | Max interdict cost |
| `--minr` | `500` | Min goal reward |
| `--maxr` | `1500` | Max goal reward |
| `--high_costs` | off | Use paper's "high" cost preset (overrides individual cost/reward flags) |
| `--validate` | off | Validate CCG against exact bilevel MIP |
| `--draw` | off | Draw the attack graph before solving |
| `--save_plot PATH` | – | Save convergence plot to file |
| `--no_plot` | off | Skip the convergence plot |
| `--solver_msg` | off | Show Gurobi solver output |

### Reproduce the paper's experiments

The experiment grid matches Table 2 of the paper (50–200 node graphs, ≈ 2.15 × nodes arcs, paper budget ranges).

```bash
# Quick subset (~2–5 min)
uv run python experiments/run_experiments.py --quick

# Full experiment
uv run python experiments/run_experiments.py

# Use paper's "high" cost preset
uv run python experiments/run_experiments.py --high_costs

# Save results to CSV
uv run python experiments/run_experiments.py --csv results.csv
```

### Verify the graph generator

```bash
uv run python data/generator.py
```

---

## Graph generation (Section 5 of the paper)

Graphs are generated following the four-step method from the paper:

1. Nodes divided into `L` levels; vulnerability nodes at level 1, goal nodes at level `L`.
   A single synthetic source (level 0) connects to all level-1 nodes with attack cost 0
   and an effectively infinite interdict cost (models unblockable entry points).
2. Each level-`l` node is connected to at least one node at level `l+1` (connectivity guarantee).
3. Random inter-level arcs are added until the target count `d` is reached.
4. Goal nodes receive rewards ~ Uniform[`minr`, `maxr`]; other arcs sample costs from their respective ranges.

**Cost/reward presets (Table 2):**

| Preset | `mina`/`maxa` | `mind`/`maxd` | `minr`/`maxr` |
|--------|---------------|---------------|----------------|
| Low (default) | 10 / 30 | 10 / 30 | 500 / 1500 |
| High | 30 / 50 | 30 / 50 | 1000 / 2000 |

**Typical parameter combinations (Table 2):**

| Nodes | L | W | d | B\_def | B\_att |
|-------|---|---|---|--------|--------|
| ≈ 50  | 5 | 10 | 100 | 75  | 125 |
| ≈ 50  | 7 | 7  | 100 | 75  | 125 |
| ≈ 100 | 5 | 20 | 197 | 150 | 150 |
| ≈ 150 | 5 | 30 | 295 | 275 | 325 |
| ≈ 200 | 5 | 40 | 393 | 375 | 425 |

---

## Solvers

| File | Description |
|------|-------------|
| `model/new_no_callbacks.py` | **Benders no-callbacks** — sequential Benders decomposition loop (Gurobi). Simpler to follow, less efficient. |
| `model/new_callbacks.py` | **Benders with callbacks** — lazy Benders cuts injected directly into Gurobi's B&B tree. Fastest on large instances. |

## Project structure

```
ThreatCut/
├── main.py                        # Single-instance entry point (runs all 3 solvers)
├── pyproject.toml                 # Dependencies and project metadata
├── data/
│   └── generator.py               # Synthetic graph generation (paper Section 5)
├── model/
│   ├── attack_graph.py            # AttackGraph data structure
│   ├── new_no_callbacks.py        # Benders decomposition — sequential loop (Gurobi)
│   └── new_callbacks.py           # Benders decomposition — lazy callbacks (Gurobi)
└── experiments/
    └── run_experiments.py         # Full computational experiment (paper Table 2 grid)
```
