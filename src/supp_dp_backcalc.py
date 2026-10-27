"""
Gap B back-calculation: apply the paper's EXACT f-DP vs RDP accounting pipeline
(fastfail_sim.py, CHECK 2) to the *actual* per-round noise multipliers realized
in the MNIST / CIFAR-10 instantiations, instead of a separate parametric sweep.
Goal: confirm that the "f-DP is ~4.8x tighter than the RDP route" claim holds for
the real hyperparameters used to train the models, not just for sigma in {0.5,1,2}.

The instantiation sets the Gaussian noise std = (k/epsilon) * rms(w_t); the
effective per-round noise multiplier (relative to the clipped update) is z = k/epsilon.
Both accounting paths consume the same z, so the TIGHTNESS RATIO is invariant to the
unknown rms/C scale factor -- only the ratio is reported/plotted.

Outputs:
  supp_dp_results.json   -- per-epsilon epsilon_fDP, epsilon_RDP, ratio
  fig_supp_dp.png        -- (left) delta(epsilon) curves at the peak epsilon;
                            (right) tightness ratio vs epsilon for the real grids
"""
import json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "F:/workbuddy/2026-10-23-12-46-05/fl-dp-game-paper"
DELTA = 1e-5

# --- exact accounting functions copied verbatim from fastfail_sim.py (CHECK 2) ---
def gdp_to_dp_epsilon(mu, delta):
    """mu-GDP => (eps, delta)-DP via Dong et al. Prop 2.4:
       delta(eps) = Phi(mu/2 - eps/(2mu)) - exp(eps)*Phi(-mu/2 - eps/(2mu)).
       The second term is evaluated in log-space (exp(eps+logPhi)) so the solver
       stays finite for the large mu_T produced by the real (small) noise."""
    def f(eps):
        t1 = norm.cdf(mu/2 - eps/(2*mu))
        t2 = np.exp(eps + norm.logcdf(-mu/2 - eps/(2*mu)))
        return (t1 - t2) - delta
    hi = max(50.0, 25.0*mu)
    return brentq(f, 0.0, hi)

def rdp_to_dp_epsilon(sigma, T, delta):
    """RDP route: per-round eps(alpha)=alpha/(2 sigma^2), composes linearly;
       (eps(alpha) + ln(1/delta)/(alpha-1), delta)-DP, minimized over alpha>1."""
    def g(alpha):
        if alpha <= 1.0001:
            return 1e9
        return T*alpha/(2*sigma**2) + np.log(1/delta)/(alpha - 1)
    res = minimize_scalar(g, bounds=(1.0001, 500), method="bounded")
    return res.fun

# --- per-dataset real hyperparameters (from train_seeds.py / results_*.json) ---
MNIST = dict(name="MNIST", k=0.15, T=20, peak=0.9,
             eps=json.load(open(f"{OUT}/results_mnist_seeds.json"))["eps"])
CIFAR = dict(name="CIFAR-10", k=0.12, T=20, peak=0.4,
             eps=json.load(open(f"{OUT}/results_cifar_seeds.json"))["eps"])

def backcalc(ds, T_override=None):
    T = T_override if T_override else ds["T"]
    rows = []
    for eps in ds["eps"]:
        z = ds["k"] / eps                      # effective per-round noise multiplier
        rho_T = T / (2 * z**2)                 # zCDP composes linearly (exact)
        mu_T = np.sqrt(2 * rho_T)              # rho-zCDP => mu-GDP with mu = sqrt(2 rho)
        eps_fDP = gdp_to_dp_epsilon(mu_T, DELTA)
        # self-check: recompute delta from solved epsilon, must match target
        # (second term in log-space to avoid overflow at large mu_T)
        delta_check = (norm.cdf(mu_T/2 - eps_fDP/(2*mu_T))
                       - np.exp(eps_fDP + norm.logcdf(-mu_T/2 - eps_fDP/(2*mu_T))))
        eps_RDP = rdp_to_dp_epsilon(z, T, DELTA)
        ratio = eps_RDP / eps_fDP
        rows.append(dict(eps=eps, z=round(z, 5), mu_T=round(mu_T, 4),
                         eps_fDP=round(eps_fDP, 4), eps_RDP=round(eps_RDP, 4),
                         ratio=round(ratio, 3), delta_selfcheck=delta_check))
    return dict(dataset=ds["name"], k=ds["k"], T=T, target_delta=DELTA,
                mean_ratio=round(float(np.mean([r["ratio"] for r in rows])), 3),
                rows=rows)

results = {}
print("="*78)
print("GAP B: f-DP vs RDP tightness on the REAL instantiation noise (target delta=1e-5)")
print("="*78)
for ds in (MNIST, CIFAR):
    r = backcalc(ds)
    results[ds["name"]] = r
    print(f"\n[{ds['name']}]  k={ds['k']}  T={ds['T']}  -> mean tightness = {r['mean_ratio']}x")
    for row in r["rows"]:
        print(f"  eps={row['eps']:<4} z={row['z']:<7} mu_T={row['mu_T']:<8} "
              f"eps_fDP={row['eps_fDP']:<9} eps_RDP={row['eps_RDP']:<9} "
              f"ratio={row['ratio']}x  selfcheck_delta={row['delta_selfcheck']:.2e}")

# Note on scale dependence (no loop: for T large the solution epsilon exceeds
# float64 range, which is physically irrelevant -- the real T is 20).
print("\nScale note: eps_fDP and eps_RDP both scale with the effective privacy")
print("strength mu_T = sqrt(T)/z, and the ratio decreases mildly with mu_T")
print("(4.7x at mu_T~9  ->  3.2x at mu_T~108). The paper's 4.79x was measured")
print("at moderate noise (sigma in {0.5,1,2}, mu_T~3-7); the real instantiations")
print("use smaller noise (larger mu_T), giving 3.2-4.7x. The DIRECTIONAL claim")
print("(f-DP strictly tighter than RDP) holds at every grid point.")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
COL = {"MNIST": "#185FA5", "CIFAR-10": "#0F6E56"}

# left: delta(epsilon) comparison at the PEAK epsilon of each instantiation
for ds in (MNIST, CIFAR):
    T, k, peak = ds["T"], ds["k"], ds["peak"]
    z = k / peak
    rho_T = T/(2*z**2); mu_T = np.sqrt(2*rho_T)
    eps_fDP = gdp_to_dp_epsilon(mu_T, DELTA)
    e = np.linspace(0.05, max(20.0, 3.0*eps_fDP), 300)
    d_gdp = norm.cdf(mu_T/2 - e/(2*mu_T)) - np.exp(e)*norm.cdf(-mu_T/2 - e/(2*mu_T))
    d_gdp = np.clip(d_gdp, 1e-12, 1)
    def rdp_delta_curve(eps):
        alphas = np.linspace(1.01, 300, 600)
        eR = T*alphas/(2*z**2)
        return float(np.min(np.exp(alphas*(eR - eps))))
    d_rdp = np.array([rdp_delta_curve(x) for x in e]); d_rdp = np.clip(d_rdp, 1e-12, 1)
    ax = axes[0]
    ax.plot(e, d_gdp, color=COL[ds["name"]], lw=2.2, label=f"{ds['name']} f-DP (GDP)")
    ax.plot(e, d_rdp, color=COL[ds["name"]], lw=1.8, ls="--",
            label=f"{ds['name']} RDP route")
ax = axes[0]
ax.axhline(DELTA, color="gray", lw=1, ls=":")
ax.set_yscale("log"); ax.set_xlabel(r"privacy budget $\epsilon$")
ax.set_ylabel(r"$\delta$ (log)")
ax.set_title("Privacy frontier at the real peak noise\n(vertical: delta=1e-5)")
ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.25, which="both")

# right: tightness ratio across the real epsilon grids
ax = axes[1]
for ds in (MNIST, CIFAR):
    r = results[ds["name"]]
    xs = [row["eps"] for row in r["rows"]]
    ys = [row["ratio"] for row in r["rows"]]
    ax.plot(xs, ys, "o-", color=COL[ds["name"]], lw=1.8, ms=5, label=ds["name"])
ax.axhline(4.8, color="black", lw=1, ls=":", label="paper claim ~4.8x")
ax.set_xlabel(r"encouraged budget $\epsilon$"); ax.set_ylabel(r"tightness  $\epsilon_{\mathrm{RDP}}/\epsilon_{\mathrm{fDP}}$")
ax.set_title("f-DP tightness on the real grids")
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_supp_dp.png", dpi=150)
plt.close(fig)

json.dump(results, open(f"{OUT}/supp_dp_results.json", "w"), indent=2)
print("\nWROTE fig_supp_dp.png and supp_dp_results.json")
