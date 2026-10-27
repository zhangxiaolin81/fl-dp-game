"""Probe: MNIST + logistic regression, does accuracy respond to epsilon (noise) and reach a useful range?"""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

np.random.seed(0); torch.manual_seed(0)
ROOT = os.path.dirname(os.path.abspath(__file__))
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
tr = datasets.MNIST(os.path.join(ROOT,"data"), train=True, download=True, transform=tfm)
te = datasets.MNIST(os.path.join(ROOT,"data"), train=False, download=True, transform=tfm)
N=4000
idx=np.random.permutation(len(tr))[:N]
trs=Subset(tr,idx)
targets=np.array([tr.targets[i] for i in idx])
labels=np.unique(targets)
NC=8; ALPHA=0.5
ci=[[] for _ in range(NC)]
for lab in labels:
    pos=np.where(targets==lab)[0]
    cnt=np.random.dirichlet([ALPHA]*NC)*len(pos); cnt=np.round(cnt).astype(int); cnt[0]+=len(pos)-cnt.sum()
    s=0
    for c in range(NC):
        ci[c].extend(pos[s:s+cnt[c]].tolist()); s+=cnt[c]
rng=np.random.default_rng(7); tau=rng.uniform(0.4,3.5,NC)
print("tau",np.round(tau,2))

class LR(nn.Module):
    def __init__(s): super().__init__(); s.W=nn.Linear(28*28,10)
    def forward(s,x): return s.W(x.flatten(1))
def loader(c):
    ds=[trs[i] for i in ci[c]]
    if not ds: return None
    return DataLoader(ds, batch_size=64, shuffle=True)
tl=DataLoader(te, batch_size=256)

def run(eps, k=0.4, R=15, seed=0):
    torch.manual_seed(seed)
    m=LR().to("cpu"); sm=torch.optim.SGD(m.parameters(),lr=0.1)
    sig=k/eps
    last=0
    for r in range(R):
        parts=[c for c in range(NC) if tau[c]>=eps]
        if not parts: break
        agg=[torch.zeros_like(p) for p in m.parameters()]
        for c in parts:
            ld=loader(c)
            if ld is None: continue
            mc=LR().to("cpu"); mc.load_state_dict(m.state_dict())
            opt=torch.optim.SGD(mc.parameters(),lr=0.1)
            mc.train()
            for x,y in ld:
                opt.zero_grad(); F.cross_entropy(mc(x),y).backward(); opt.step()
            d=[gp.detach()-lp.detach() for gp,lp in zip(m.parameters(),mc.parameters())]
            flat=torch.cat([t.flatten() for t in d])
            n=torch.norm(flat); 
            if n>1: flat=flat*(1/n)
            flat=flat+torch.randn_like(flat)*sig
            off=0
            for j,p in enumerate(m.parameters()):
                nn_=p.numel(); agg[j]+=flat[off:off+nn_].view(p.shape); off+=nn_
        for j,p in enumerate(m.parameters()):
            p.data.add_(agg[j]/len(parts))
        m.eval(); cor=0; tot=0
        with torch.no_grad():
            for x,y in tl:
                cor+=(m(x).argmax(1)==y).sum().item(); tot+=y.size(0)
        last=cor/tot
    return last

for eps in [0.3,1.0,2.5]:
    a=run(eps)
    print(f"eps={eps} sigma={0.4/eps:.3f} acc={a:.3f} part={np.mean([tau[c]>=eps for c in range(NC)]):.2f}")
print("done")
