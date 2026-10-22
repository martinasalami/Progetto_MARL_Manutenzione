# Robotic Cell Maintenance with Q-Learning

A small **single-agent Reinforcement Learning** project developed for the [Multi-Agent Reinforcement Learning PhD course](https://github.com/smarianimore/2026-phd-marl) at the University of Modena and Reggio Emilia.

## Overview

The project simulates a robotic cell subject to wear and failures. An agent learns when to **produce** or **perform maintenance**, balancing productivity and breakdown risk.

- **States:** Healthy, Worn, Critical
- **Actions:** Produce, Maintain
- **Algorithm:** tabular Q-learning with epsilon-greedy exploration
- **Evaluation:** comparison with always-produce, periodic-maintenance and random policies

The environment and its parameters are synthetic and intended for educational use.

## Repository Structure

```text
Manutenzione_Cella/manutenzione_rl/manutenzione_rl.py   # Simulation, training and evaluation
risultati_man_e*/                                       # Saved experiments and plots
README.md
```

## Run

Requires Python and Matplotlib. From the repository root:

```bash
python -m pip install matplotlib
python Manutenzione_Cella/manutenzione_rl/manutenzione_rl.py
```

The script trains and evaluates the agent, compares it with the baseline strategies, and saves **CSV results, configuration files and plots** in a folder named after the experiment settings. Parameters can be changed at the beginning of the Python script.

## References

- [Course repository](https://github.com/smarianimore/2026-phd-marl)
- [Official laboratory examples](https://github.com/smarianimore/2026-dai-marl-lab)
