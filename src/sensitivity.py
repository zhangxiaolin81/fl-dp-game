"""
P1-3: sensitivity ablation of the privacy-budget threshold delta*.

Two complementary parts:

(A) THEORETICAL REGIME MAP (fast, no training)
    delta*(gamma, a) with uniform tolerance U[a,b], b=3.5:
        delta* = sqrt(gamma*(gamma+b)) - gamma   if gamma > a^2/(b-2a)   [interior]
               = a                                otherwise              [boundary]
    We plot delta* over (gamma, a) and overlay the regime boundary
    gamma0(a) = a^2/(b-2a). The two empirical localisations (MNIST interior,
    CIFAR-10 boundary) are marked from the gamma fitted off the real curves.

(B) EMPIRICAL gamma-FIT + PREDICTION CHECK (fast, from existing results)
    Fit the accuracy-saturation parameter gamma from the empirical
    acc(eps) curves via least squares to  acc(eps) ~= A * eps/(eps+gamma)
    (A = nonDP ceiling). Then predict delta* via the regime formula and
    compare with the empirical argmax delta*_emp. This demonstrates that
    delta* is fully determined by the fitted (gamma, a) and is therefore
    sensitive to every knob that shifts gamma (noise scale k) or a
    (participation structure: N, non-IID alpha).

Run:
    python sensitivity.py theory      -> fig_regime_map.png (+ prints fits)
    python sensitivity.py fit         -> prints MNIST/CIFAR gamma & delta* pred
"""
import os, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.abspath(__file__))
B = 3.5  # upper bound of uniform tolerance U[a, b]


# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------
def delta_int(gamma, b=B):
    return np.sqrt(gamma * (gamma + b)) - gamma

def regime_boundary(a, b=B):
    """gamma0(a): for gamma > gamma0 -> interior; else boundary."""
    return a * a / (b - 2 * a)

def delta_star(gamma, a, b=B):
    if a >= b / 2:
        return a  # degenerate: saturation edge already at/above midpoint
    g0 = regime_boundary(a, b)
    if gamma > g0:
        return delta_int(gamma, b)
    return a


# ---------------------------------------------------------------------------
# gamma fit from an empirical acc(eps) curve
# ---------------------------------------------------------------------------
def fit_gamma(eps, acc, A):
    """acc(eps) ~= A * eps/(eps+gamma); least-squares scalar gamma>0."""
    eps = np.asarray(eps, float); acc = np.asarray(acc, float)
    def loss(g):
        return np.sum((acc - A * eps / (eps + g)) ** 2)
    res = minimize_scalar(loss, bounds=(1e-3, 8.0), method="bounded")
    return res.x

def load_results(path):
    return json.load(open(os.path.join(ROOT, path)))


# ---------------------------------------------------------------------------
# (A) theoretical regime map
# ---------------------------------------------------------------------------
def theory_map():
    # mark empirical points by fitting gamma from the real curves
    rm = load_results("results_mnist.json")
    rc = load_results("results_cifar.json")
    g_mn = fit_gamma(rm["eps"], rm["acc"], rm["nonDP_final"])
    g_cf = fit_gamma(rc["eps"], rc["acc"], rc["nonDP_final"])
    # a = min privacy-tolerance ceiling (participation-saturation edge).
    # NOTE: each dataset needs its OWN fresh Generator (default_rng is stateful).
    a_mn = float(np.random.default_rng(7).uniform(0.4, 3.5, 8).min())
    a_cf = float(np.random.default_rng(7).uniform(0.4, 3.5, 8).min())
    d_mn_pred = delta_star(g_mn, a_mn)
    d_cf_pred = delta_star(g_cf, a_cf)
    d_mn_emp = rm["delta_star"]
    d_cf_emp = rc["delta_star"]

    ga = np.linspace(0.2, 1.2, 300)        # a (participation boundary)
    gg = np.linspace(0.01, 1.2, 300)       # gamma (accuracy elasticity)
    AA, GG = np.meshgrid(ga, gg)
    DD = np.zeros_like(AA)
    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
            DD[i, j] = delta_star(GG[i, j], AA[i, j])

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    cs = ax.contourf(AA, GG, DD, levels=20, cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax); cbar.set_label(r"$\delta^*$ (optimal budget)")
    # regime boundary gamma0(a)
    am = ga[ga < B / 2]
    ax.plot(am, regime_boundary(am, B), color="white", lw=2.2, ls="--",
            label=r"regime boundary $\gamma_0(a)=a^2/(b-2a)$")
    ax.text(0.42, 0.92, "interior regime\n($\\delta^*$ interior peak)",
            color="white", fontsize=10, ha="center", fontweight="bold")
    ax.text(0.92, 0.10, "boundary regime\n($\\delta^*=a$)",
            color="white", fontsize=10, ha="center", fontweight="bold")
    # empirical markers
    ax.scatter([a_mn], [g_mn], color="#ff4d4d", s=120, zorder=6, marker="*",
               edgecolor="k",
               label=rf"MNIST fit: $\gamma$={g_mn:.2f}, emp $\delta^*$={d_mn_emp:.2f}")
    ax.scatter([a_cf], [g_cf], color="#ffd24d", s=120, zorder=6, marker="*",
               edgecolor="k",
               label=rf"CIFAR-10 fit: $\gamma$={g_cf:.2f}, emp $\delta^*$={d_cf_emp:.2f}")
    ax.set_xlabel(r"participation-saturation boundary $a=\min_i\tau_i$")
    ax.set_ylabel(r"accuracy-elasticity parameter $\gamma$")
    ax.set_title(r"Sensitivity of $\delta^*$ to $(\gamma,a)$: two regimes of the same law")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "fig_regime_map.png"), dpi=150)
    plt.close(fig)
    print(f"[fit] MNIST : gamma={g_mn:.3f} a={a_mn:.3f} -> delta*_pred={d_mn_pred:.3f} "
          f"(emp={rm['delta_star']:.2f})")
    print(f"[fit] CIFAR : gamma={g_cf:.3f} a={a_cf:.3f} -> delta*_pred={d_cf_pred:.3f} "
          f"(emp={rc['delta_star']:.2f})")
    print("[map] saved fig_regime_map.png")


# ---------------------------------------------------------------------------
# (B) empirical gamma-fit + prediction check (print only)
# ---------------------------------------------------------------------------
def fit_check():
    rm = load_results("results_mnist.json")
    rc = load_results("results_cifar.json")
    a_mn = float(np.random.default_rng(7).uniform(0.4, 3.5, 8).min())
    a_cf = float(np.random.default_rng(7).uniform(0.4, 3.5, 8).min())
    for name, R, a in [("MNIST", rm, a_mn), ("CIFAR", rc, a_cf)]:
        g = fit_gamma(R["eps"], R["acc"], R["nonDP_final"])
        d_pred = delta_star(g, a)
        d_emp = R["delta_star"]
        print(f"{name:7s}: A={R['nonDP_final']:.3f} gamma={g:.3f} a={a:.3f} "
              f"delta*_pred={d_pred:.3f} delta*_emp={d_emp:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "theory"
    if cmd == "theory":
        theory_map()
    elif cmd == "fit":
        fit_check()
    else:
        print("usage: sensitivity.py [theory|fit]")
