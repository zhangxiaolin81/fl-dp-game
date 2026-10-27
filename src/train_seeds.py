"""
Multi-seed robustness runner for the FL + DP game paper (Task P1-1).

Produces mean +/- std of the final test accuracy (and the induced system
utility U(eps)=participation*accuracy) across S training seeds, with the
DATA PARTITION and the privacy-tolerance ceilings tau_c held FIXED (only the
SGD / data-shuffle seed varies). This isolates the optimisation variance,
which is the dominant stochasticity in the federated training loop.

The point estimates (seed 0) exactly reproduce the single-seed curves already
reported in results_mnist.json / results_cifar.json, so the new error bars
overlay the existing figures without changing the central claim.

Usage:
  python train_seeds.py mnist [SEEDS]   -> results_mnist_seeds.json
  python train_seeds.py cifar [SEEDS]   -> results_cifar_seeds.json

The script is resumable: completed (eps, seed) pairs are written to disk
incrementally and skipped on restart.
"""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = sys.argv[1] if len(sys.argv) > 1 else "mnist"
NSEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEEDS = list(range(NSEEDS))


# ---------------------------------------------------------------------------
# MNIST instantiation (logistic regression, fixed Gaussian noise sigma=K/eps)
# ---------------------------------------------------------------------------
def build_mnist():
    NOISE_K = 0.15; NC = 8; ALPHA = 0.5; R = 20; EPOCHS = 1; N = 4000
    ETA = 0.1; BATCH = 64
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(os.path.join(ROOT, "data"), train=True, download=True, transform=tfm)
    te = datasets.MNIST(os.path.join(ROOT, "data"), train=False, download=True, transform=tfm)
    np.random.seed(0)  # FIXED partition across seeds
    idx = np.random.permutation(len(tr))[:N]; trs = Subset(tr, idx)
    targets = np.array([tr.targets[i] for i in idx]); labels = np.unique(targets)
    ci = [[] for _ in range(NC)]
    for lab in labels:
        pos = np.where(targets == lab)[0]
        cnt = np.round(np.random.dirichlet([ALPHA] * NC) * len(pos)).astype(int)
        cnt[0] += len(pos) - cnt.sum()
        s = 0
        for c in range(NC):
            ci[c].extend(pos[s:s + cnt[c]].tolist()); s += cnt[c]
    rng = np.random.default_rng(7); tau = rng.uniform(0.4, 3.5, NC)  # FIXED tolerances

    class LR(nn.Module):
        def __init__(s): super().__init__(); s.W = nn.Linear(28 * 28, 10)
        def forward(s, x): return s.W(x.flatten(1))
    def loader(c):
        ds = [trs[i] for i in ci[c]]
        if not ds: return None
        return DataLoader(ds, batch_size=BATCH, shuffle=True)
    tl = DataLoader(te, batch_size=256)
    def eval_acc(m):
        m.eval(); cor = tot = 0
        with torch.no_grad():
            for x, y in tl: cor += (m(x.flatten(1)).argmax(1) == y).sum().item(); tot += y.size(0)
        return cor / max(tot, 1)
    def client_train(m0, c):
        mc = LR(); mc.load_state_dict(m0.state_dict())
        opt = torch.optim.SGD(mc.parameters(), lr=ETA)
        ld = loader(c)
        if ld is None: return None
        mc.train()
        for _ in range(EPOCHS):
            for x, y in ld:
                opt.zero_grad(); F.cross_entropy(mc(x.flatten(1)), y).backward(); opt.step()
        return [p.detach().clone() for p in mc.parameters()]
    def run(eps, nonDP=False, seed=0):
        torch.manual_seed(seed)
        m = LR(); sig = 0.0 if nonDP else NOISE_K / eps; accs = []
        for r in range(R):
            parts = [c for c in range(NC) if (nonDP or tau[c] >= eps)]
            if not parts: accs.append(eval_acc(m)); continue
            cp = [client_train(m, c) for c in parts]; cp = [x for x in cp if x is not None]
            if not cp: accs.append(eval_acc(m)); continue
            np_ = len(cp[0]); new = []
            for j in range(np_):
                avg = torch.stack([x[j] for x in cp], 0).mean(0)
                if sig > 0: avg = avg + torch.randn_like(avg) * sig
                new.append(avg)
            gp = list(m.parameters())
            with torch.no_grad():
                for j, p in enumerate(gp): p.copy_(new[j])
            accs.append(eval_acc(m))
        return accs

    eps_grid = [float(x) for x in np.round(np.linspace(0.3, 3.0, 10), 3)]
    return dict(run=run, eps_grid=eps_grid, tau=tau, NC=NC, R=R,
                res_path=os.path.join(ROOT, "results_mnist_seeds.json"),
                nonDP_final=None)


# ---------------------------------------------------------------------------
# CIFAR-10 instantiation (small CNN, scale-free Gaussian noise sigma=(K/eps)*rms)
# ---------------------------------------------------------------------------
def build_cifar():
    N, R, EPOCHS, BATCH, NC = 20000, 20, 5, 64, 8
    ALPHA = 0.5; ETA = 0.12; K = 0.12
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                   (0.2023, 0.1994, 0.2010))])
    cifar_root = os.path.join(ROOT, "data_cifar")
    extracted = os.path.join(cifar_root, "cifar-10-batches-py")
    if not os.path.isdir(extracted):
        local_tar = os.path.join(ROOT, "cifar-10-python.tar.gz")
        if os.path.isfile(local_tar):
            import tarfile
            with tarfile.open(local_tar) as tf: tf.extractall(cifar_root)
    tr = datasets.CIFAR10(cifar_root, train=True, download=False, transform=tfm)
    te = datasets.CIFAR10(cifar_root, train=False, download=False, transform=tfm)
    np.random.seed(0)  # FIXED partition across seeds
    idx = np.random.RandomState(0).permutation(len(tr))[:N]; trs = Subset(tr, idx)
    targets = np.array([tr.targets[i] for i in idx]); labels = np.unique(targets)
    ci = [[] for _ in range(NC)]
    for lab in labels:
        pos = np.where(targets == lab)[0]
        cnt = np.round(np.random.dirichlet([ALPHA] * NC) * len(pos)).astype(int)
        cnt[0] += len(pos) - cnt.sum()
        s = 0
        for c in range(NC):
            ci[c].extend(pos[s:s + cnt[c]].tolist()); s += cnt[c]
    rng = np.random.default_rng(7); tau = rng.uniform(0.4, 3.5, NC)  # FIXED tolerances

    class CNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.c1 = nn.Conv2d(3, 16, 3, padding=1); s.c2 = nn.Conv2d(16, 32, 3, padding=1)
            s.fc1 = nn.Linear(32 * 8 * 8, 64); s.fc2 = nn.Linear(64, 10)
        def forward(s, x):
            x = F.relu(s.c1(x)); x = F.max_pool2d(x, 2)
            x = F.relu(s.c2(x)); x = F.max_pool2d(x, 2)
            x = x.flatten(1); x = F.relu(s.fc1(x)); return s.fc2(x)
    def loader(c):
        ds = [trs[i] for i in ci[c]]
        if not ds: return None
        return DataLoader(ds, batch_size=BATCH, shuffle=True)
    tl = DataLoader(te, batch_size=256)
    def eval_acc(m):
        m.eval(); cor = tot = 0
        with torch.no_grad():
            for x, y in tl: cor += (m(x).argmax(1) == y).sum().item(); tot += y.size(0)
        return cor / max(tot, 1)
    def client_train(m0, c):
        mc = CNN(); mc.load_state_dict(m0.state_dict())
        opt = torch.optim.SGD(mc.parameters(), lr=ETA, momentum=0.0, weight_decay=0.0)
        ld = loader(c)
        if ld is None: return None
        mc.train()
        for _ in range(EPOCHS):
            for x, y in ld:
                opt.zero_grad(); F.cross_entropy(mc(x), y).backward(); opt.step()
        return [p.detach().clone() for p in mc.parameters()]
    def run(eps, nonDP=False, seed=0):
        torch.manual_seed(seed)
        m = CNN(); accs = []
        for r in range(R):
            parts = [c for c in range(NC) if (nonDP or tau[c] >= eps)]
            if not parts: accs.append(eval_acc(m)); continue
            cp = [client_train(m, c) for c in parts]; cp = [x for x in cp if x is not None]
            if not cp: accs.append(eval_acc(m)); continue
            np_ = len(cp[0]); new = []
            for j in range(np_):
                avg = torch.stack([x[j] for x in cp], 0).mean(0)
                if not nonDP:
                    gparms = list(m.parameters())
                    rms = torch.stack([p.norm() for p in gparms]).mean().item() / \
                          np.sqrt(sum(p.numel() for p in gparms) / len(gparms))
                    sig = (K / eps) * rms
                    avg = avg + torch.randn_like(avg) * sig
                new.append(avg)
            gp = list(m.parameters())
            with torch.no_grad():
                for j, p in enumerate(gp): p.copy_(new[j])
            accs.append(eval_acc(m))
        return accs

    eps_grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.9, 1.7, 2.9]
    return dict(run=run, eps_grid=eps_grid, tau=tau, NC=NC, R=R,
                res_path=os.path.join(ROOT, "results_cifar_seeds.json"),
                nonDP_final=None)


# ---------------------------------------------------------------------------
# Resumable multi-seed driver
# ---------------------------------------------------------------------------
def load_res(path, eps_grid, nseeds):
    if os.path.isfile(path):
        try:
            r = json.load(open(path))
            if r.get("eps") == [float(x) for x in eps_grid] and len(r.get("acc", [])) == len(eps_grid):
                # backfill shape if seeds count changed
                if len(r["acc"][0]) != nseeds:
                    return None
                return r
        except Exception:
            pass
    return {"eps": [float(x) for x in eps_grid], "seeds": NSEEDS,
            "acc": [[None] * nseeds for _ in range(len(eps_grid))],
            "nonDP": [None] * nseeds}


def main():
    if DATASET == "mnist":
        C = build_mnist()
    elif DATASET == "cifar":
        C = build_cifar()
    else:
        raise SystemExit("usage: train_seeds.py [mnist|cifar] [NSEEDS]")
    run, eps_grid, tau, NC, R, res_path = (C["run"], C["eps_grid"], C["tau"],
                                           C["NC"], C["R"], C["res_path"])
    res = load_res(res_path, eps_grid, NSEEDS)
    print(f"[cfg] dataset={DATASET} eps={np.round(eps_grid,2)} seeds={SEEDS} tau={np.round(tau,2)}")

    # nonDP baseline variance (should be ~constant; recorded for completeness)
    for s in SEEDS:
        if res["nonDP"][s] is None:
            t0 = time.time(); a = run(10.0, nonDP=True)[-1]; res["nonDP"][s] = float(a)
            json.dump(res, open(res_path, "w"), indent=2)
            print(f"[nonDP] seed={s} acc={a:.3f} t={time.time()-t0:.1f}s")
    nf = float(np.mean([x for x in res["nonDP"] if x is not None]))
    print(f"[nonDP] mean={nf:.3f}")

    done_total = sum(1 for row in res["acc"] for v in row if v is not None)
    total = len(eps_grid) * NSEEDS
    for ei, eps in enumerate(eps_grid):
        for si, s in enumerate(SEEDS):
            if res["acc"][ei][si] is not None:
                continue
            t0 = time.time(); fa = run(eps, seed=s)[-1]
            res["acc"][ei][si] = float(fa)
            json.dump(res, open(res_path, "w"), indent=2)
            done_total += 1
            print(f"[{done_total}/{total}] eps={eps:.2f} seed={s} acc={fa:.3f} t={time.time()-t0:.1f}s")

    # aggregate
    acc_mean, acc_std, part, U_mean, U_std = [], [], [], [], []
    for ei, eps in enumerate(eps_grid):
        row = [v for v in res["acc"][ei] if v is not None]
        am = float(np.mean(row)); asd = float(np.std(row, ddof=1)) if len(row) > 1 else 0.0
        p = float(np.mean([tau[c] >= eps for c in range(NC)]))
        acc_mean.append(am); acc_std.append(asd); part.append(p)
        U_mean.append(p * am); U_std.append(p * asd)
    res["acc_mean"] = acc_mean; res["acc_std"] = acc_std
    res["part"] = part; res["U_mean"] = U_mean; res["U_std"] = U_std
    res["nonDP_final"] = nf
    bi = int(np.argmax(U_mean)); res["delta_star"] = eps_grid[bi]; res["U_star"] = U_mean[bi]
    json.dump(res, open(res_path, "w"), indent=2)
    print(f"DONE dataset={DATASET} delta*={res['delta_star']:.2f} U*={U_mean[bi]:.3f} "
          f"(+/-{U_std[bi]:.3f}) over {len(SEEDS)} seeds")


if __name__ == "__main__":
    main()
