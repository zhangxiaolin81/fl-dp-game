"""Regenerate figures from results.json with corrected LaTeX titles."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "results.json")))
sw = R["threshold_sweep"]
pu = R["privacy_utility"]
conv = R["convergence"]
best = R["delta_star_emp"]

# 2 privacy-utility (fixed title)
plt.figure(figsize=(6,4))
plt.plot(pu["eps_gdp"], pu["acc"], marker='s', color="#0F6E56", label="Our (f-DP accounting)")
plt.plot(pu["eps_zcdp"], pu["acc"], marker='^', color="#E07B39", label="Same mechanism, zCDP accounting")
plt.xlabel(r"Reported $\epsilon$ at $\delta=10^{-5}$"); plt.ylabel("Test accuracy")
plt.title(r"Privacy-Utility trade-off (f-DP tighter $\Rightarrow$ larger allowed $\epsilon$)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(ROOT,"fig_pu.png"), dpi=130); plt.close()

# 1 convergence
plt.figure(figsize=(6,4))
for k, col in [("nonDP_FedAvg","#888"),("uniformDP_eps2","#E07B39"),("game_at_delta","#185FA5")]:
    plt.plot(range(1,len(conv[k])+1), conv[k], marker='o', color=col, label=k)
plt.xlabel("Round"); plt.ylabel("Test accuracy"); plt.title("Convergence (CIFAR-10, 8 clients)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(ROOT,"fig_conv.png"), dpi=130); plt.close()

# 3 threshold
xs=[d["eps_enc"] for d in sw]
plt.figure(figsize=(6,4))
plt.plot(xs,[d["U"] for d in sw], marker='o', color="#534AB7", label=r"System utility $U(\epsilon)$")
plt.plot(xs,[d["part_rate"] for d in sw],'--',color="#993C1D", label="Participation rate")
plt.plot(xs,[d["final_acc"] for d in sw],':',color="#0F6E56", label="Accuracy")
plt.axvline(best, color="red", ls='-', alpha=0.7, label=f"$\\delta^*={best:.2f}$")
plt.xlabel(r"Encouraged privacy budget $\epsilon$"); plt.ylabel("Utility / rate / accuracy")
plt.title("Phase-transition threshold on real training")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(ROOT,"fig_thresh.png"), dpi=130); plt.close()
print("figures regenerated")
