# Game-Theoretic Incentive Mechanism for Personalized Differential Privacy in Federated Learning

Reproduction package for the paper:

> **Game-Theoretic Incentive Mechanism for Personalized Differential Privacy in Federated Learning: Privacy Budget Threshold Optimization under Tight f-DP Accounting**
> Zhang Xiaolin, Lu Lei. Submitted to *Future Generation Computer Systems* (FGCS).

The paper models the server-client interaction in federated learning as a Stackelberg game under tight f-DP (Gaussian Differential Privacy) accounting, derives the phase-transition threshold $\delta^*$ of the privacy budget, and adds a verifiable budget mechanism based on black-box privacy auditing.

## Repository structure

```
fl-dp-game/
├── src/
│   ├── fastfail_sim.py            # §7.1  Parametric fast-fail verification of δ*
│   ├── train_fl.py                # §7.2  FL training loop (MNIST, non-IID)
│   ├── train_fl2.py               # §7.2  FL training loop (variant)
│   ├── train_seeds.py             # §7.2–7.3  Multi-seed (5 seeds) MNIST/CIFAR-10 runs
│   ├── train_cifar.py             # §7.3  CIFAR-10 large-scale instantiation
│   ├── verify_sim.py              # §7.4  Verifiable budget mechanism simulation
│   ├── sensitivity.py             # §7.5  Sensitivity to k, α, N_c
│   ├── supp_stackelberg_game.py   # §7.6  Full-game Stackelberg equilibrium (8 clients)
│   ├── supp_dp_backcalc.py        # §7.6  f-DP accounting on the real per-round noise
│   ├── plot_robust.py             # figure generation (robustness/supplementary panels)
│   ├── plot_only.py               # figure generation (standalone)
│   └── mnist_probe.py             # dataset probe utility
└── results/
    ├── results.json               # §7.1  fast-fail results
    ├── results_mnist.json         # §7.2  MNIST single-run results
    ├── results_mnist_seeds.json   # §7.2  MNIST 5-seed results
    ├── results_cifar.json         # §7.3  CIFAR-10 single-run results
    ├── results_cifar_seeds.json   # §7.3  CIFAR-10 5-seed results
    ├── verify_results.json        # §7.4  verifiable mechanism results
    ├── sensitivity.json           # §7.5  sensitivity sweep results
    ├── supp_game_results.json     # §7.6  full-game equilibrium results
    └── supp_dp_results.json       # §7.6  f-DP back-calculation results
```

## Requirements

- Python 3.10+
- numpy, scipy, matplotlib
- torch, torchvision (only needed for the training scripts `train_*.py`)

```bash
pip install -r requirements.txt
```

All experiments run on a **single CPU thread** (deterministic) and complete within a day on a standard desktop machine. The parametric scripts (`fastfail_sim.py`, `verify_sim.py`, `sensitivity.py`, `supp_*.py`) finish in minutes.

## Reproducing the paper

| Paper section | Command |
|---|---|
| §7.1 Fast-fail verification | `python src/fastfail_sim.py` |
| §7.2–7.3 Multi-seed training (MNIST / CIFAR-10) | `python src/train_seeds.py` |
| §7.4 Verifiable budget mechanism | `python src/verify_sim.py` |
| §7.5 Sensitivity sweeps | `python src/sensitivity.py` |
| §7.6 Full-game equilibrium | `python src/supp_stackelberg_game.py` |
| §7.6 f-DP vs RDP back-calculation | `python src/supp_dp_backcalc.py` |

Each script writes its result JSON into `results/` and prints the figures referenced in the paper. The committed `results/*.json` are exactly the files used to produce the figures and numbers in the submission.

## Data

All experimental data are derived from the **public MNIST and CIFAR-10 benchmarks** (downloaded automatically by `torchvision` on first run) plus synthetic parametric verification. No proprietary or personal data are used.

## Key results

- Phase-transition threshold δ\* ≈ 1.18 (uniform tolerance) / ≈ 0.70 (log-normal); measured multi-seed localisation δ\* ≈ 0.90 (MNIST, interior regime) and ≈ 0.40 (CIFAR-10, boundary regime).
- f-DP composition is empirically 3–5× tighter than the RDP/zCDP path at the target δ = 10⁻⁵, an advantage that decreases monotonically with privacy strength.
- The full-game equilibrium lies safely below the reduced-proxy threshold, reproducing both regimes.

## License

Released under the MIT License.

## Citation

If you use this code, please cite the paper (citation block to be updated upon acceptance).
