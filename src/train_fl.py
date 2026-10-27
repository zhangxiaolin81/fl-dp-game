"""
Real FL experiment harness for the game-theoretic incentive paper.
Instantiates the threshold model on CIFAR-10:
  - N clients, each with a privacy tolerance tau_c (heterogeneous).
  - Server encourages a budget epsilon_enc; a client participates iff tau_c >= epsilon_enc.
  - Participating clients train with Gaussian DP-SGD noise calibrated to epsilon_enc
    under f-DP (mu = C/sigma = epsilon_enc). Larger epsilon_enc -> less noise -> higher
    accuracy, but lower participation. System utility U = participation_rate * accuracy
    is non-monotonic -> empirical phase-transition threshold delta*.
Outputs: results.json + fig_conv.png, fig_pu.png, fig_thresh.png
"""
import os, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0); torch.manual_seed(0)
DEVICE = "cpu"
ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------- data ----------------
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
def load_cifar():
    tr = datasets.CIFAR10(os.path.join(ROOT, "data"), train=True, download=True, transform=tfm)
    te = datasets.CIFAR10(os.path.join(ROOT, "data"), train=False, download=True, transform=tfm)
    return tr, te

try:
    train_full, test_set = load_cifar()
    DATASET = "CIFAR-10"
    N_TRAIN = 4000
    idx = np.random.permutation(len(train_full))[:N_TRAIN]
    train_sub = Subset(train_full, idx)
except Exception as e:
    print("CIFAR-10 download failed:", e, "-> using synthetic fallback")
    DATASET = "SYNTHETIC"
    train_sub = None; test_set = None

def make_synthetic(n_train=4000, n_test=800, dim=32*32*3, n_class=4, seed=1):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (dim, n_class)).astype(np.float32)
    def gen(n):
        y = rng.integers(0, n_class, n)
        X = (np.dot(W, np.eye(n_class)[y].T).T + rng.normal(0, 0.8, (n, dim))).reshape(n, 3, 32, 32).astype(np.float32)
        X = (X - X.mean((0,2,3), keepdims=True)) / (X.std((0,2,3), keepdims=True) + 1e-6)
        return torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))
    return gen(n_train), gen(n_test)

if DATASET == "SYNTHETIC":
    (Xtr, ytr), (Xte, yte) = make_synthetic()
    train_sub = list(zip(Xtr, ytr))
    test_set = list(zip(Xte, yte))

# ---------------- partitioning (heterogeneous) ----------------
N_CLIENTS = 8
ALPHA = 0.5
if DATASET == "CIFAR-10":
    targets = np.array([train_full.targets[i] for i in idx])
    labels = np.unique(targets)
    client_idx = [[] for _ in range(N_CLIENTS)]
    for lab in labels:
        lab_pos = np.where(targets == lab)[0]
        counts = np.random.dirichlet([ALPHA] * N_CLIENTS) * len(lab_pos)
        counts = np.round(counts).astype(int)
        # fix rounding
        diff = len(lab_pos) - counts.sum()
        counts[0] += diff
        start = 0
        for c in range(N_CLIENTS):
            client_idx[c].extend(lab_pos[start:start+counts[c]].tolist())
            start += counts[c]
else:
    perm = np.random.permutation(len(train_sub))
    client_idx = [perm[c::N_CLIENTS].tolist() for c in range(N_CLIENTS)]

# privacy tolerance per client (heterogeneous, fixed)
rng = np.random.default_rng(7)
tau = rng.uniform(0.4, 3.5, size=N_CLIENTS)

# ---------------- model ----------------
class TinyCNN(nn.Module):
    def __init__(self, n_class=10 if DATASET != "SYNTHETIC" else 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1); self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1); self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32*8*8, n_class)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x))); x = F.max_pool2d(x, 2)
        x = x.flatten(1); return self.fc(x)

N_CLASS = 10 if DATASET != "SYNTHETIC" else 4
model = TinyCNN(N_CLASS).to(DEVICE)
CW = sum(p.numel() for p in model.parameters())
print(f"Dataset={DATASET} clients={N_CLIENTS} model_params={CW} tau={np.round(tau,2)}")

# ---------------- helpers ----------------
def get_client_loader(c):
    if DATASET == "CIFAR-10":
        ds = Subset(train_full, client_idx[c])
    else:
        ds = [train_sub[i] for i in client_idx[c]]
    if len(ds) == 0:
        return None
    bs = min(64, len(ds))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)

if DATASET == "CIFAR-10":
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
else:
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

def clip_and_noise(delta, sigma, C=1.0):
    # delta: flat parameter tensor; clip to C, add Gaussian noise scaled by sigma*C
    norm = torch.norm(delta).item()
    if norm > C:
        delta = delta * (C / norm)
    if sigma > 0:
        delta = delta + torch.randn_like(delta) * (sigma * C)
    return delta

def accuracy(model, loader):
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item(); total += y.size(0)
    return correct / max(total, 1)

# f-DP accounting
def gdp_to_dp(mu, eps):
    # returns delta for given mu-GDP and eps (eq gdp2dp)
    return math.erf((mu/2 - eps/(2*mu))/math.sqrt(2)) - math.exp(eps)*math.erf((-mu/2 - eps/(2*mu))/math.sqrt(2))
    # note: erf form of Phi
def phi(z): return 0.5*(1+math.erf(z/math.sqrt(2)))
def gdp_delta(mu, eps):
    return phi(mu/2 - eps/(2*mu)) - math.exp(eps)*phi(-mu/2 - eps/(2*mu))
def zcdp_delta(rho, eps):
    return phi(-eps/(2*math.sqrt(rho)) + math.sqrt(rho)) - math.exp(eps)*phi(-eps/(2*math.sqrt(rho)) - math.sqrt(rho))

# ---------------- training loop ----------------
R = 10  # rounds
LR = 0.05
C = 1.0

def run_fl(epsilon_enc, sigma_override=None, rounds=R, seed=0):
    torch.manual_seed(seed)
    m = TinyCNN(N_CLASS).to(DEVICE)
    w0 = [p.detach().clone() for p in m.parameters()]
    test_accs = []
    participants_history = []
    for r in range(rounds):
        # participation: tau_c >= epsilon_enc
        parts = [c for c in range(N_CLIENTS) if tau[c] >= epsilon_enc]
        participants_history.append(len(parts))
        if len(parts) == 0:
            test_accs.append(accuracy(m, test_loader)); continue
        # mu = epsilon_enc (GDP); sigma = C/mu
        mu = epsilon_enc if sigma_override is None else (C/sigma_override)
        sigma = C / mu if sigma_override is None else sigma_override
        flat_shapes = [p.shape for p in m.parameters()]
        agg = [torch.zeros_like(p) for p in m.parameters()]
        for c in parts:
            loader = get_client_loader(c)
            if loader is None: continue
            mc = TinyCNN(N_CLASS).to(DEVICE)
            for lp, gp in zip(mc.parameters(), m.parameters()): lp.data.copy_(gp.data)
            opt = torch.optim.SGD(mc.parameters(), lr=LR)
            mc.train()
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(); loss = F.cross_entropy(mc(x), y); loss.backward(); opt.step()
            # delta
            deltas = [gp.detach().flatten() - lp.detach().flatten() for gp, lp in zip(m.parameters(), mc.parameters())]
            flat = torch.cat([d.flatten() for d in deltas])
            flat = clip_and_noise(flat, sigma, C)
            # redistribute
            off = 0
            for j, p in enumerate(m.parameters()):
                n = p.numel()
                blk = flat[off:off+n].view(p.shape)
                agg[j] += blk
                off += n
        for j, p in enumerate(m.parameters()):
            p.data.copy_(p.data + agg[j] / len(parts))
        test_accs.append(accuracy(m, test_loader))
    return test_accs, participants_history

# ---------------- sweep over encouraged budget ----------------
eps_grid = np.round(np.linspace(0.3, 3.5, 16), 3)
results = {"dataset": DATASET, "clients": N_CLIENTS, "rounds": R, "tau": tau.tolist(), "eps_grid": eps_grid.tolist(), "threshold_sweep": []}
for eps in eps_grid:
    t0 = time.time()
    accs, parts = run_fl(eps, seed=0)
    final_acc = accs[-1]
    part_rate = np.mean(parts) / N_CLIENTS
    U = part_rate * final_acc
    # f-DP reporting: mu = eps ; pick delta target 1e-5 -> solve eps_gdp? report (eps, delta) at delta=1e-5
    # recompute: total over R rounds: mu_tot = sqrt(R)*mu ; then (eps_out, 1e-5)
    mu_tot = math.sqrt(R) * eps
    # find eps such that gdp_delta(mu_tot, eps)=1e-5
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo+hi)/2
        if gdp_delta(mu_tot, mid) > 1e-5: lo = mid
        else: hi = mid
    eps_reported = (lo+hi)/2
    # zCDP comparison
    rho = (mu_tot**2)/2
    lo2, hi2 = 0.0, 20.0
    for _ in range(60):
        mid = (lo2+hi2)/2
        if zcdp_delta(rho, mid) > 1e-5: lo2 = mid
        else: hi2 = mid
    eps_zcdp = (lo2+hi2)/2
    results["threshold_sweep"].append({
        "eps_enc": float(eps), "final_acc": float(final_acc),
        "part_rate": float(part_rate), "U": float(U),
        "eps_gdp_1e-5": float(eps_reported), "eps_zcdp_1e-5": float(eps_zcdp),
    })
    print(f"eps={eps:.2f} acc={final_acc:.3f} part={part_rate:.2f} U={U:.3f} "
          f"epsGDP={eps_reported:.3f} epsZCDP={eps_zcdp:.3f} t={time.time()-t0:.1f}s")

# empirical threshold
best = max(results["threshold_sweep"], key=lambda d: d["U"])
results["delta_star_emp"] = best["eps_enc"]
results["delta_star_U"] = best["U"]

# ---------------- convergence comparison ----------------
conv = {}
conv["nonDP_FedAvg"] = run_fl(10.0, sigma_override=0.0, seed=0)[0]   # no noise
conv["uniformDP_eps2"] = run_fl(2.0, seed=0)[0]
conv["game_at_delta"] = run_fl(best["eps_enc"], seed=0)[0]
results["convergence"] = {k: v for k, v in conv.items()}

# ---------------- privacy-utility curve ----------------
pu = {"eps_enc": [], "acc": [], "eps_gdp": [], "eps_zcdp": []}
for d in results["threshold_sweep"]:
    pu["eps_enc"].append(d["eps_enc"]); pu["acc"].append(d["final_acc"])
    pu["eps_gdp"].append(d["eps_gdp_1e-5"]); pu["eps_zcdp"].append(d["eps_zcdp_1e-5"])
results["privacy_utility"] = pu

with open(os.path.join(ROOT, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ---------------- figures ----------------
# 1 convergence
plt.figure(figsize=(6,4))
for k, col in [("nonDP_FedAvg","#888"),("uniformDP_eps2","#E07B39"),("game_at_delta","#185FA5")]:
    plt.plot(range(1,R+1), conv[k], marker='o', color=col, label=k)
plt.xlabel("Round"); plt.ylabel("Test accuracy"); plt.title("Convergence (CIFAR-10, 8 clients)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(ROOT,"fig_conv.png"), dpi=130); plt.close()

# 2 privacy-utility
plt.figure(figsize=(6,4))
plt.plot(pu["eps_gdp"], pu["acc"], marker='s', color="#0F6E56", label="Our (f-DP accounting)")
plt.plot(pu["eps_zcdp"], pu["acc"], marker='^', color="#E07B39", label="Same mechanism, zCDP accounting")
plt.xlabel(r"Reported $\epsilon$ at $\delta=10^{-5}$"); plt.ylabel("Test accuracy")
plt.title(r"Privacy–Utility trade-off (f-DP tighter $\Rightarrow$ larger allowed $\epsilon$)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(ROOT,"fig_pu.png"), dpi=130); plt.close()

# 3 threshold
sw = results["threshold_sweep"]
xs = [d["eps_enc"] for d in sw]
plt.figure(figsize=(6,4))
plt.plot(xs, [d["U"] for d in sw], marker='o', color="#534AB7", label="System utility $U(\\epsilon)$")
plt.plot(xs, [d["part_rate"] for d in sw], '--', color="#993C1D", label="Participation rate")
plt.plot(xs, [d["final_acc"] for d in sw], ':', color="#0F6E56", label="Accuracy")
plt.axvline(best["eps_enc"], color="red", ls='-', alpha=0.7, label=f"$\\delta^*={best['eps_enc']:.2f}$")
plt.xlabel("Encouraged privacy budget $\\epsilon$"); plt.ylabel("Utility / rate / accuracy")
plt.title("Phase-transition threshold on real training")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(ROOT,"fig_thresh.png"), dpi=130); plt.close()

print("DONE. delta*_emp =", best["eps_enc"], "U=", round(best["U"],3))
