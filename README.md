<h1>🔰 ThreatCut</h1>
<h3><em>Graph Interdiction & Cybersecurity Optimization Framework</em></h3>

![Blue Team Badge](https://img.shields.io/badge/Team-Blue-blue) 

This repository contains a Python implementation of the bi-level optimization framework presented in the research article _"Interdicting attack graphs to protect organizations from cyber attacks: a bi-level defender-attacker model"_.

The goal of the project is to preproduce and extend the methodology proposed in the paper, providing a computational tool for analyzing how defensive investments can optimally disrupt an attacker's ability to compromise a system.

---

## Why ThreatCut?

Zero-day vulnerabilities can emerge at any moment in software running on any device.  
A single user can fall for a phishing attempt.  
A misconfigured IoT sensor, legacy server, or forgotten endpoint can become an attacker's entry point.

Modern cyber attacks often follow multi-step intrusion paths that can be represented as **attack graphs**.

**ThreatCut** was created to:  
* measure how far an attacker could propagate inside a network,  
* estimate the potential damage,  
* evaluate the **defensive cost** required to limit or stop that propagation.

Using **graph interdiction models** and **MILP optimization**, ThreatCut provides a structured way to analyze attack impact and defense strategies.

---

## 📜 License

MIT License.

---


