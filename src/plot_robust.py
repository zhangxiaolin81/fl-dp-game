"""
Regenerate figures for the robustness section (P1-1 error bars + P1-3 sensitivity).

  - MNIST  : fig_pu.png, fig_thresh.png  with mean +/- std over training seeds
  - CIFAR  : fig_cifar_pu.png, fig_cifar_thresh.png  with mean +/- std
  - P1-3   : fig_sensitivity.png  (delta* and fitted gamma vs k / alpha / N)

Reads:
  results_mnist.json + results_mnist_seeds.json
  results_cifar.json + results_cifar_seeds.json
  sensitivity.json   (empirical ablation)

Each block is guarded: if the required file is missing it is skipped, so the
script can be run incrementally as the background jobs complete.

Usage: python plot_robust.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))

def p(*a): return os.path.join(ROOT, *a)


# ---------------------------------------------------------------------------
# P1-1: error bars on privacy-utility and threshold figures
# ---------------------------------------------------------------------------
def errorbar_figs(rp_point, rp_seeds, f_pu, f_thresh, title_ds, color_acc, color_part):
    if not (os.path.isfile(rp_point) and os.path.isfile(rp_seeds)):
        print(f"[skip] missing {rp_point} or {rp_seeds}")
        return
    R = json.load(open(rp_point))
    S = json.load(open(rp_seeds))
    if "acc_mean" not in S:
        print(f"[skip] {rp_seeds} still running (no aggregated stats yet)")
        return
    eps = R["eps"]; part = R["part"]
    am = S["acc_mean"]; ast = S["acc_std"]; um = S["U_mean"]; ust = S["U_std"]
    ds = R["delta_star"]

    # privacy-utility
    plt.figure(figsize=(6, 4))
    plt.errorbar(eps, am, yerr=ast, marker='s', color=color_acc, capsize=3,
                 elinewidth=1.2, label=r"Accuracy (mean$\pm$std)")
    plt.plot(eps, part, marker='^', color=color_part, label="Participation rate")
    plt.xlabel(r"Encouraged privacy budget $\epsilon$"); plt.ylabel("Accuracy / participation")
    plt.title(f"Privacy-Utility on {title_ds} (multi-seed)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(p(f_pu), dpi=130); plt.close()

    # threshold
    plt.figure(figsize=(6, 4))
    plt.errorbar(eps, um, yerr=ust, marker='o', color="#534AB7", capsize=3,
                 elinewidth=1.2, label=r"System utility $U(\epsilon)$ (mean$\pm$std)")
    plt.axvline(ds, color="red", ls='-', alpha=0.7, label=f"$\\delta^*={ds:.2f}$")
    plt.xlabel(r"Encouraged privacy budget $\epsilon$"); plt.ylabel("System utility")
    plt.title(f"Empirical phase-transition threshold on {title_ds}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(p(f_thresh), dpi=130); plt.close()
    print(f"[ok] {title_ds}: error-bar figures written ({f_pu}, {f_thresh})")


# ---------------------------------------------------------------------------
# P1-3: sensitivity of delta* to k / alpha / N
# ---------------------------------------------------------------------------
def sensitivity_fig():
    if not os.path.isfile(p("sensitivity.json")):
        print("[skip] missing sensitivity.json")
        return
    D = json.load(open(p("sensitivity.json")))
    by_id = {d["id"]: d for d in D}

    def grab(prefix):
        return [(d["id"].split("=")[1] if "=" in d["id"] else "baseline",
                 d["delta_star_emp"], d["gamma_fit"], d["regime"])
                for d in D if d["id"].startswith(prefix) or d["id"] == "baseline"]

    # build three families
    families = []
    # K-sweep
    ks = []
    for cid in ["baseline", "K=0.08", "K=0.25"]:
        if cid in by_id:
            d = by_id[cid]; ks.append((d["K"], d["delta_star_emp"], d["gamma_fit"], d["regime"]))
    ks = sorted(ks)
    # alpha-sweep
    al = []
    for cid in ["baseline", "alpha=0.1", "alpha=1.0"]:
        if cid in by_id:
            d = by_id[cid]; al.append((d["ALPHA"], d["delta_star_emp"], d["gamma_fit"], d["regime"]))
    al = sorted(al)
    # N-sweep
    ns = []
    for cid in ["NC=4", "baseline", "NC=16"]:
        if cid in by_id:
            d = by_id[cid]; ns.append((d["NC"], d["delta_star_emp"], d["gamma_fit"], d["regime"]))
    ns = sorted(ns)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    specs = [("Noise scale  k  (Gaussian std $\\propto k/\\epsilon$)", ks, "k"),
             ("Non-IID concentration  $\\alpha$  (lower = more skewed)", al, "alpha"),
             ("Number of clients  N  (per-client data fixed)", ns, "N")]
    for ax, (title, fam, xlab) in zip(axes, specs):
        if not fam:
            ax.set_title(title + "\n(no data)"); continue
        xs = [f[0] for f in fam]
        de = [f[1] for f in fam]
        ga = [f[2] for f in fam]
        cols = ["#C0392B" if f[3] == "interior" else "#2E6FB0" for f in fam]
        ax.plot(xs, de, '-o', color="#534AB7", zorder=1)
        for x, y, c in zip(xs, de, cols):
            ax.scatter([x], [y], s=90, color=c, zorder=3, edgecolor="k")
        ax.set_xticks(xs)
        ax2 = ax.twinx()
        ax2.plot(xs, ga, '--s', color="#0F6E56", alpha=0.7, label=r"fitted $\gamma$")
        ax2.set_ylabel(r"fitted $\gamma$", color="#0F6E56")
        ax2.tick_params(axis='y', labelcolor="#0F6E56")
        ax.set_xlabel(xlab); ax.set_ylabel(r"empirical $\delta^*$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    # regime colour legend on the first panel
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor="#C0392B",
                      markeredgecolor="k", markersize=9, label="interior regime"),
               Line2D([0], [0], marker='o', color='w', markerfacecolor="#2E6FB0",
                      markeredgecolor="k", markersize=9, label="boundary regime"),
               Line2D([0], [0], color="#534AB7", marker='o', label=r"empirical $\delta^*$"),
               Line2D([0], [0], color="#0F6E56", ls='--', marker='s',
                      label=r"fitted $\gamma$ (right axis)")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.suptitle(r"Sensitivity of the empirical threshold $\delta^*$ to deployment knobs "
                 "(red = interior regime, blue = boundary regime)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(p("fig_sensitivity.png"), dpi=150); plt.close(fig)
    print("[ok] fig_sensitivity.png written")


if __name__ == "__main__":
    errorbar_figs("results_mnist.json", "results_mnist_seeds.json",
                  "fig_pu.png", "fig_thresh.png", "MNIST", "#0F6E56", "#E07B39")
    errorbar_figs("results_cifar.json", "results_cifar_seeds.json",
                  "fig_cifar_pu.png", "fig_cifar_thresh.png", "CIFAR-10", "#0F6E56", "#E07B39")
    sensitivity_fig()
    print("plot_robust done")
