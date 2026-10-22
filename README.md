# RL for Robotic Cell Maintenance

A tabular Q-learning project for condition-based maintenance scheduling in a simulated industrial robotic cell.

Developed as a software project for the *Multi-Agent Reinforcement Learning* PhD course at the University of Modena and Reggio Emilia. **Single-agent RL** case study: one agent decides whether to continue production or perform maintenance.

## Overview

The environment represents a robotic cell subject to stochastic wear and failures. At each decision step, the agent observes the current health condition and chooses an action to balance production, maintenance costs, and breakdown risks.

| Component | Definition |
| --- | --- |
| States | `Healthy`, `Worn`, `Critical` |
| Actions | `Produce`, `Maintain` |
| Failure probabilities when producing | 1%, 7%, 38% (by state) |
| Rewards | Successful production: +5; maintenance: -3.5; failure: -18 |
| Episode length | 60 decision steps |

Maintenance restores the cell to `Healthy`. A failure also resets it to `Healthy` after a simplified corrective repair. All transition probabilities and rewards are synthetic assumptions, not measured industrial data.

## Learning and Evaluation

The project implements model-free, off-policy Q-learning with an exponentially decaying epsilon-greedy exploration policy. Three agents are trained independently (seeds `11`, `23`, and `37`; learning rate `0.12`; discount factor `0.95`).

The learned greedy policies are evaluated over **300 test episodes per agent**, without exploration or parameter updates, and compared with three non-learning strategies:

- **Always Produce** — continuous production without preventive maintenance
- **Periodic 8** — maintenance every eight steps
- **Random** — random selection between production and maintenance

Evaluation includes mean cumulative reward, failures, maintenance actions, produced units, and failure-free episode rate. The repository also contains learning curves, Q-table visualisations, and operational comparisons.

## Repository Structure

```text
.
├── README.md
├── Manutenzione_Cella/manutenzione_rl/
│   └── manutenzione_rl.py      # Environment, Q-learning, evaluation and plots
├── risultati_manutenzione_2500/
├── risultati_manutenzione_5000/
├── risultati_manutenzione_50000/
└── risultati_manutenzione_50000_e1750/
```

Each saved experiment folder contains `config.json`, evaluation and Q-table CSV files, training history, and four PNG figures.

## Run the Experiment

Requires **Python** and **Matplotlib**. From the repository root:

```bash
python -m pip install matplotlib
python Manutenzione_Cella/manutenzione_rl/manutenzione_rl.py
```

The current script runs 2500 training episodes per seed, displays the generated plots, and writes the new outputs to `risultati_manutenzione/` in the current working directory. Training settings are defined near the top of `manutenzione_rl.py`.

## Results and Scope

For the saved **2,500-episode** experiment, mean evaluation results are:

| Strategy | Mean reward | Mean failures |
| --- | ---: | ---: |
| Always Produce | 174.57 | 5.45 |
| Periodic 8 | 176.18 | 2.80 |
| Random | 29.08 | 0.64 |
| **Q-learning** | **220.97** | **1.31** |

The saved results also include 5,000- and 50,000-episode experiments, with an alternative epsilon-decay schedule for the latter. The results illustrate the trade-off between productivity, maintenance frequency, and failure risk.

This is a **simulation-based condition-based maintenance** study, not a predictive-maintenance system. Results do not demonstrate global optimality or performance on a real robotic cell.

## Course Reference

[Multi-Agent Reinforcement Learning — PhD course](https://github.com/smarianimore/2026-phd-marl) · [Official laboratory examples](https://github.com/smarianimore/2026-dai-marl-lab)
