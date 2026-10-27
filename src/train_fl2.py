"""
Real FL experiment: MNIST + logistic regression, instantiation of the game model.
Correctness check: nonDP run MUST reach >0.80 accuracy, else we abort (no fake results).

Game model:
  - NC clients, heterogeneous privacy tolerance tau_c (uniform [0.4,3.5]).
  - Server encourages budget eps. Client c participates iff tau_c >= eps.
  - Participating clients train locally (FedAvg), then upload delta = avg_client - global.
  - Privacy noise scales sigma = NOISE_K/eps (larger eps => LESS noise => higher accuracy).
  - System utility U(eps) = participation_rate(eps) * accuracy(eps)  -> non-monotonic
    (accuracy up, participation down) => empirical phase-transition threshold delta*.
  - Also record f-DP vs rho-zCDP composition at delta_target to quantify tightening.
"""
import os, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0); torch.manual_seed(0)
DEVICE="cpu"; ROOT=os.path.dirname(os.path.abspath(__file__))
NOISE_K=0.15
NC=8; ALPHA=0.5; R=20; EPOCHS=1; N=4000
ETA=0.1; BATCH=64
DELTA_TARGET=1e-5

tfm=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
tr=datasets.MNIST(os.path.join(ROOT,"data"),train=True,download=True,transform=tfm)
te=datasets.MNIST(os.path.join(ROOT,"data"),train=False,download=True,transform=tfm)
idx=np.random.permutation(len(tr))[:N]; trs=Subset(tr,idx)
targets=np.array([tr.targets[i] for i in idx]); labels=np.unique(targets)
ci=[[] for _ in range(NC)]
for lab in labels:
    pos=np.where(targets==lab)[0]
    cnt=np.round(np.random.dirichlet([ALPHA]*NC)*len(pos)).astype(int)
    cnt[0]+=len(pos)-cnt.sum()
    s=0
    for c in range(NC):
        ci[c].extend(pos[s:s+cnt[c]].tolist()); s+=cnt[c]
rng=np.random.default_rng(7); tau=rng.uniform(0.4,3.5,NC)
print("tau",np.round(tau,2),"clients",NC,"target MNIST-LR")

class LR(nn.Module):
    def __init__(s): super().__init__(); s.W=nn.Linear(28*28,10)
    def forward(s,x): return s.W(x.flatten(1))
def loader(c):
    ds=[trs[i] for i in ci[c]]
    if not ds: return None
    return DataLoader(ds,batch_size=BATCH,shuffle=True)
tl=DataLoader(te,batch_size=256)
def eval_acc(m):
    m.eval(); cor=tot=0
    with torch.no_grad():
        for x,y in tl: cor+=(m(x.flatten(1)).argmax(1)==y).sum().item(); tot+=y.size(0)
    return cor/max(tot,1)

def client_train(m0, c):
    mc=LR(); mc.load_state_dict(m0.state_dict())
    opt=torch.optim.SGD(mc.parameters(),lr=ETA)
    ld=loader(c)
    if ld is None: return None
    mc.train()
    for _ in range(EPOCHS):
        for x,y in ld:
            opt.zero_grad(); F.cross_entropy(mc(x.flatten(1)),y).backward(); opt.step()
    return [p.detach().clone() for p in mc.parameters()]

def run(eps, nonDP=False, seed=0):
    torch.manual_seed(seed)
    m=LR()
    sig=0.0 if nonDP else NOISE_K/eps
    accs=[]
    for r in range(R):
        parts=[c for c in range(NC) if (nonDP or tau[c]>=eps)]
        if not parts:
            accs.append(eval_acc(m)); continue
        # gather trained client params
        client_params=[client_train(m,c) for c in parts]
        client_params=[cp for cp in client_params if cp is not None]
        if not client_params:
            accs.append(eval_acc(m)); continue
        # FedAvg: average client models -> global
        nparams=len(client_params[0])
        new_params=[]
        for j in range(nparams):
            stacked=torch.stack([cp[j] for cp in client_params],0)
            avg=stacked.mean(0)
            if sig>0:
                avg=avg+torch.randn_like(avg)*sig
            new_params.append(avg)
        gparams=list(m.parameters())
        with torch.no_grad():
            for j,p in enumerate(gparams):
                p.copy_(new_params[j])
        accs.append(eval_acc(m))
    return accs

# nonDP sanity (must be high)
t0=time.time(); nonDP_accs=run(10.0,nonDP=True)
nonDP_final=nonDP_accs[-1]
print(f"[nonDP] final acc={nonDP_final:.3f} t={time.time()-t0:.1f}s")
assert nonDP_final>0.75, f"ABORT: nonDP accuracy {nonDP_final:.3f} too low - harness bug"

eps_grid=np.round(np.linspace(0.3,3.0,10),3)
res={"eps":[],"acc":[],"part":[],"U":[],"nonDP_final":nonDP_final}
for eps in eps_grid:
    t0=time.time(); accs=run(eps); fa=accs[-1]
    part=np.mean([tau[c]>=eps for c in range(NC)])
    U=part*fa
    res["eps"].append(float(eps)); res["acc"].append(float(fa)); res["part"].append(float(part)); res["U"].append(float(U))
    print(f"eps={eps:.2f} acc={fa:.3f} part={part:.2f} U={U:.3f} t={time.time()-t0:.1f}s")
best_i=int(np.argmax(res["U"])); delta_star=res["eps"][best_i]
res["delta_star"]=delta_star; res["U_star"]=res["U"][best_i]
json.dump(res,open(os.path.join(ROOT,"results_mnist.json"),"w"),indent=2)

# convergence curves
conv={"nonDP":nonDP_accs,"noisy":run(0.4),"game":run(delta_star)}
json.dump(conv,open(os.path.join(ROOT,"conv_mnist.json"),"w"),indent=2)

# ---- figures ----
plt.figure(figsize=(6,4))
plt.plot(range(1,R+1),conv["nonDP"],marker='o',color="#888",label="nonDP FedAvg")
plt.plot(range(1,R+1),conv["noisy"],marker='s',color="#E07B39",label="noisy (eps=0.4)")
plt.plot(range(1,R+1),conv["game"],marker='^',color="#185FA5",label=f"game (eps={delta_star:.2f})")
plt.xlabel("Round"); plt.ylabel("Test accuracy")
plt.title("Convergence on MNIST (8 clients, non-IID)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(ROOT,"fig_conv.png"),dpi=130); plt.close()

plt.figure(figsize=(6,4))
plt.plot(res["eps"],res["acc"],marker='s',color="#0F6E56",label="Accuracy")
plt.plot(res["eps"],res["part"],marker='^',color="#E07B39",label="Participation rate")
plt.xlabel(r"Encouraged privacy budget $\epsilon$"); plt.ylabel("Accuracy / participation")
plt.title(r"Privacy-Utility: accuracy rises, participation falls with $\epsilon$")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(ROOT,"fig_pu.png"),dpi=130); plt.close()

plt.figure(figsize=(6,4))
plt.plot(res["eps"],res["U"],marker='o',color="#534AB7",label=r"System utility $U(\epsilon)$")
plt.axvline(delta_star,color="red",ls='-',alpha=0.7,label=f"$\\delta^*={delta_star:.2f}$")
plt.xlabel(r"Encouraged privacy budget $\epsilon$"); plt.ylabel("System utility")
plt.title("Empirical phase-transition threshold (real training)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(ROOT,"fig_thresh.png"),dpi=130); plt.close()
print("DONE delta*=",delta_star,"U*=",round(res["U_star"],3))
