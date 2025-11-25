<h1>🔰 ThreatCut</h1>
<h3><em>Graph Interdiction & Cybersecurity Optimization Framework</em></h3>

![Blue Team Badge](https://img.shields.io/badge/Team-Blue-blue) 

---

## Why ThreatCut?

No network can ever be **fully secure**.

Zero-day vulnerabilities can emerge at any moment in software running on any device.  
A single user can fall for a phishing attempt.  
A misconfigured IoT sensor, legacy server, or forgotten endpoint can become an attacker's entry point.

In practice, **any device can be the first compromised node**.

**ThreatCut** was created to study exactly this:  
* to measure how far an attacker could propagate inside a network,  
* to estimate the potential damage,  
* nd to evaluate the **defensive cost** required to limit or stop that propagation.

Using **graph interdiction models** and **MILP optimization**, ThreatCut provides a structured way to analyze attack impact and defense strategies.

---

## ✨ Features

- Optimal node/edge interdiction under budget constraints  
- MILP-based defensive strategy computation (OR-Tools)  
- Modular architecture for research and experimentation  
- Extensible with heuristics or learning-based approaches  

---

## 📁 Project Structure

ThreatCut/

├── data/             # Graphs, datasets

├── model/             # MILP formulation

├── solvers/           # Solver interfaces

├── utils/             # Helpers, graph tools

└── experiments/       # Simulations and benchmarks


---

## 🛣️ Roadmap / TODO

### **1. Implement MILP baseline**  
Replicate attack-graph interdiction instances from the referenced research article: 
*Interdicting attack graphs to protect organizations from cyber attacks, Authors: Apurba K. Nandi, Hugh R. Medal, Satish Vadlamani*.  

### **2. Add heuristic solvers**  
To handle large real-world networks where MIP becomes slow:  
- greedy interdiction  
- local search  
- simulated annealing / genetic algorithms  
- hybrid MIP + heuristic warm-starts  

### **3. Dynamic & stochastic extensions**  
- multi-period attack progression  
- uncertainty / zero-day probability modeling  

### **4. ML-based criticality prediction**  
Use ML or GNNs to prioritize nodes before optimization.

---

## 📜 License

MIT License.

---

<div align="center">

### 🔰 *ThreatCut — Cutting Threats Before They Spread.*

</div>


