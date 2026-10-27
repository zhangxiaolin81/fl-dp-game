"""
Numerical experiment for the VERIFIABLE BUDGET MECHANISM (Sec. 6 of the paper).

Scenario
--------
Strategic clients may "low-noise lie" (低噪谎报): they *declare* privacy budget
epsilon_decl (so the server accounts noise sigma_h = sigma0/epsilon_decl) but
actually train with a *smaller* noise sigma_c = r * sigma_h (r < 1) to obtain
higher local accuracy while understating their true privacy loss.

The server audits each client k times (black-box auditing, Jagielski et al.
NeurIPS'20). Because the local gradient signal is (approximately) constant across
the k repetitions, the spread of the k uploads estimates the *effective* noise
scale sigma_est. The server flags a client as cheating when
    sigma_est < (1 - tau) * sigma_h .
With more repetitions k, the estimator variance shrinks as 1/sqrt(k), so the
detectable gap Delta_k -> 0 as k grows -- validating Theorem (verifiability).

We also compare system-level compliance (privacy overspend) WITH vs WITHOUT the
verification mechanism as the cheater fraction p grows.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20261025)

SIGMA0 = 0.5          # base noise scale: sigma = SIGMA0 / epsilon
DIM = 2000           # number of model parameters (generic)
S0 = 0.10             # per-coordinate signal magnitude (constant across audits)
TAU = 0.30            # detection margin: flag if sigma_est < (1-TAU)*sigma_h
N_TRIALS = 4000

EPS_DECL = [0.5, 1.0, 2.0]
R_GRID = [0.4, 0.6, 0.8, 1.0]      # r=1.0 == honest
K_GRID = [4, 8, 16, 32, 64]


def est_sigma(epsilon_decl, r, k):
    """Return (detected_flag, sigma_est) for one audited client.

    The server audits via a single *canary* scalar response (black-box auditing,
    Jagielski et al. NeurIPS'20): it injects a marked sample and observes one
    scalar sensitivity y = s0 + noise(0, sigma_c) repeated k times. The effective
    noise scale is estimated from the k-1 degrees-of-freedom sample std, so the
    estimation error shrinks as 1/sqrt(k) -- exactly the Delta_k -> 0 claim.
    """
    sigma_h = SIGMA0 / epsilon_decl
    sigma_c = r * sigma_h
    y = rng.normal(S0, sigma_c, size=k)          # k canary observations
    sigma_est = np.std(y, ddof=1)                # sample std, df = k-1
    return (sigma_est < (1 - TAU) * sigma_h), sigma_est


def detection_rate(epsilon_decl, r, k):
    flags = 0
    for _ in range(N_TRIALS):
        flag, _ = est_sigma(epsilon_decl, r, k)
        flags += int(flag)
    return flags / N_TRIALS


# ---- (a) detection rate vs k, for several cheat gaps r ----
det_a = {}
for eps in EPS_DECL:
    det_a[eps] = {r: [detection_rate(eps, r, k) for k in K_GRID] for r in R_GRID}

# ---- (b) detection rate vs r (gap), at k=16 ----
K_FIX = 16
det_b = {eps: [detection_rate(eps, r, K_FIX) for r in R_GRID] for eps in EPS_DECL}

# ---- (c) compliance pass rate vs cheater fraction p ----
# Without verification: true budget of a cheater is epsilon_decl / r (> declared).
# Server caps total declared budget at EPS_REQ; overspend = sum of (true-declared).
# With verification: detected cheaters are penalised and revert to honest, so the
# *effective* cheater fraction is p*(1 - detection_rate(r=cheat_gap, k)).
EPS_REQ_MARGIN = 0.15   # allowed relative overspend before compliance fails
CHEAT_GAP = 0.6         # cheaters lie with r = 0.6 (i.e. 40% less noise)
K_VERIF = 16
N_CLIENTS = 20

# detection rate for the chosen gap, averaged over declaration levels
det_gap = np.mean([detection_rate(eps, CHEAT_GAP, K_VERIF) for eps in EPS_DECL])


def compliance_pass(p, with_verification):
    # p: cheater fraction
    n_cheat = int(round(p * N_CLIENTS))
    if with_verification:
        n_eff = n_cheat * (1 - det_gap)   # undetected cheaters only
    else:
        n_eff = n_cheat
    # overspend relative to declared: each cheater inflates true budget by (1/r - 1)
    overspend = n_eff * (1.0 / CHEAT_GAP - 1.0) / N_CLIENTS
    return 1.0 if overspend <= EPS_REQ_MARGIN else 0.0


P_GRID = np.linspace(0.0, 0.6, 13)
pass_no = [compliance_pass(p, False) for p in P_GRID]
pass_yes = [compliance_pass(p, True) for p in P_GRID]

# smooth pass rate by averaging over a sliding window for readability
def smooth(y, w=3):
    out = np.zeros_like(y, float)
    for i in range(len(y)):
        lo, hi = max(0, i-w), min(len(y), i+w+1)
        out[i] = np.mean(y[lo:hi])
    return out

pass_no_s = smooth(pass_no)
pass_yes_s = smooth(pass_yes)

# ============================ FIGURE ============================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
colors = {"0.4": "#b30000", "0.6": "#e34a33", "0.8": "#fc8d59", "1.0": "#878787"}

# (a)
for eps in EPS_DECL:
    for r in R_GRID:
        ls = "--" if r == 1.0 else "-"
        axes[0].plot(K_GRID, det_a[eps][r], marker="o", ls=ls,
                    color=colors[str(r)], label=f"r={r}, ε={eps}")
axes[0].axhline(1.0, color="k", lw=0.6, ls=":")
axes[0].set_xlabel("Number of audit repetitions k")
axes[0].set_ylabel("Detection rate of low-noise cheating")
axes[0].set_title("(a) Detection rate → 1 as k grows (Δ_k→0)")
axes[0].legend(fontsize=7, loc="lower right")
axes[0].grid(alpha=0.3)

# (b)
for eps in EPS_DECL:
    axes[1].plot(R_GRID, det_b[eps], marker="s", label=f"ε={eps}")
axes[1].axvline(1.0 - TAU, color="k", ls=":", lw=0.8)
axes[1].set_xlabel("Actual noise ratio r (declared = 1.0)")
axes[1].set_ylabel(f"Detection rate (k={K_FIX})")
axes[1].set_title("(b) Detectable once r < 1−τ")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# (c)
axes[2].plot(P_GRID, pass_no_s, marker="^", color="#e34a33", label="Without verification")
axes[2].plot(P_GRID, pass_yes_s, marker="o", color="#238b45", label="With verification")
axes[2].set_xlabel("Cheater fraction p among clients")
axes[2].set_ylabel("Compliance pass rate (smoothed)")
axes[2].set_title("(c) Verification sustains compliance")
axes[2].legend(fontsize=8)
axes[2].set_ylim(-0.05, 1.05)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_verify.png", dpi=150)
plt.close()

# ============================ SUMMARY ============================
summary = {
    "params": {"SIGMA0": SIGMA0, "DIM": DIM, "S0": S0, "TAU": TAU,
               "N_TRIALS": N_TRIALS, "CHEAT_GAP": CHEAT_GAP, "K_VERIF": K_VERIF},
    "detection_vs_k": det_a,
    "detection_vs_r_k16": det_b,
    "verification_detection_rate_gap": float(det_gap),
    "compliance_pass_no_verification": [float(x) for x in pass_no_s],
    "compliance_pass_with_verification": [float(x) for x in pass_yes_s],
    "P_GRID": [float(x) for x in P_GRID],
}
with open("verify_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("=== VERIFIABLE BUDGET MECHANISM — numerical summary ===")
print(f"detection rate at gap r=0.6 (k=16), averaged over ε-levels: {det_gap:.3f}")
print("detection vs k (r=0.6, ε=1.0):",
      [round(x, 3) for x in det_a[1.0][0.6]])
print("detection vs r (k=16, ε=1.0):",
      dict(zip([str(r) for r in R_GRID], [round(x, 3) for x in det_b[1.0]])))
print("compliance pass rate WITHOUT verification (p=0,.3,.6):",
      [round(pass_no_s[int(i)], 2) for i in np.linspace(0, len(P_GRID)-1, 3)])
print("compliance pass rate WITH verification    (p=0,.3,.6):",
      [round(pass_yes_s[int(i)], 2) for i in np.linspace(0, len(P_GRID)-1, 3)])
print("fig_verify.png and verify_results.json written.")
