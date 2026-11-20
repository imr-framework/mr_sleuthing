# Comprehensive meta-classifier training from cache v2 (multi-model x R{4,8} x plane{axial,sagittal}).
# (1) JOINT deployment model: all conditions + augmentation -> accurate on any input; eval per (model,plane,R).
# (2) CONTINUAL LEARNING (replay+EWC) vs naive -> anti-forgetting + generalization (paper Fig 5).
# (3) LEAVE-ONE-MODEL-OUT -> geometry-driven generalization to unseen architectures.
import os, copy, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings("ignore")
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from scipy.ndimage import rotate
dev="mps" if torch.backends.mps.is_available() else "cpu"; print("device:",dev,flush=True)
OUT=os.environ.get("MRS_OUT","outputs/meta_classifier")
Z=np.load(f"{OUT}/meta_slices_v2.npz",allow_pickle=True)
IMG=Z["imgs"]; LAB=Z["labels"]; MOD=Z["model"].astype(str); SUBJ=Z["subj"].astype(str)
SPLIT=Z["split"].astype(str); PLANE=Z["plane"].astype(str); RR=Z["R"].astype(int)
ORDER=["dcnn","zssr","ppn"]; NM={"dcnn":"A (dCNN)","zssr":"B (ZSSR)","ppn":"C (PPN)"}
PLANES={"dcnn":["axial","sagittal"],"zssr":["axial","sagittal"],"ppn":["axial"]}
RSET=sorted(set(RR.tolist()))
print("slices",len(LAB),"| reals",int((LAB==0).sum()),"recons",int((LAB==1).sum()),"| R",RSET,flush=True)
rng=np.random.default_rng(0)
norm=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
def augment(img):
    if rng.random()<0.5: img=np.fliplr(img)
    if rng.random()<0.7: img=rotate(img,float(rng.uniform(-10,10)),reshape=False,order=1,mode="nearest")
    img=np.clip(img*float(rng.uniform(0.9,1.1))+float(rng.uniform(-0.08,0.08)),0,1)   # contrast/brightness
    if rng.random()<0.5: img=np.clip(img+rng.normal(0,0.02,img.shape),0,1)             # noise
    return np.ascontiguousarray(img,dtype=np.float32)
class DS(Dataset):
    def __init__(self,idx,aug=False): self.idx=list(idx); self.aug=aug
    def __len__(self): return len(self.idx)
    def __getitem__(self,j):
        i=self.idx[j]; img=IMG[i].astype(np.float32)
        if self.aug: img=augment(img)
        return norm(torch.from_numpy(np.repeat(img[None],3,0))), int(LAB[i])
def where(model=None,split=None,plane=None,R=None,label=None):
    m=np.ones(len(LAB),bool)
    if model is not None: m&=(MOD==model)
    if split is not None: m&=(SPLIT==split)
    if plane is not None: m&=(PLANE==plane)
    if R is not None: m&=(RR==R)
    if label is not None: m&=(LAB==label)
    return np.where(m)[0]
def fresh():
    m=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); m.fc=nn.Linear(m.fc.in_features,2); return m.to(dev)
def balance_idx(idx):  # oversample the minority class (reals) so training is balanced -> preserves specificity
    idx=np.asarray(idx); lab=LAB[idx]; pos=idx[lab==1]; neg=idx[lab==0]
    if len(neg)<len(pos): neg=rng.choice(neg,len(pos),replace=True)
    elif len(pos)<len(neg): pos=rng.choice(pos,len(neg),replace=True)
    return np.concatenate([pos,neg])
def auc_on(model,idx):
    if len(idx)==0: return float("nan")
    model.eval(); P=[];Y=[]
    with torch.no_grad():
        for x,y in DataLoader(DS(idx),batch_size=64): P+=list(torch.softmax(model(x.to(dev)),1)[:,1].cpu().numpy()); Y+=list(y.numpy())
    return roc_auc_score(Y,P) if len(set(Y))>1 else float("nan")
def eval_cond(model,m,plane,R):  # reals(test,plane) [R-independent] + model recons(test,plane,R)
    neg=where(model="real",split="test",plane=plane); pos=where(model=m,split="test",plane=plane,R=R)
    return auc_on(model,np.concatenate([neg,pos]))
def train(model,idx,epochs,aug=True,replay=None,tasks=None,lam=0.0):
    tr=list(idx)+([i for i,_ in replay] if replay else [])
    opt=torch.optim.Adam(model.parameters(),lr=1e-4); crit=nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x,y in DataLoader(DS(tr,aug=aug),batch_size=32,shuffle=True):
            opt.zero_grad(); loss=crit(model(x.to(dev)),y.to(dev))
            if lam>0 and tasks:
                for star,F in tasks:
                    for n,p in model.named_parameters(): loss=loss+lam*(F[n]*(p-star[n])**2).sum()
            loss.backward(); opt.step()
    return model
def fisher(model,idx,nmax=256):
    F={n:torch.zeros_like(p) for n,p in model.named_parameters()}; crit=nn.CrossEntropyLoss(); model.eval(); c=0
    for x,y in DataLoader(DS(idx),batch_size=16,shuffle=True):
        model.zero_grad(); crit(model(x.to(dev)),y.to(dev)).backward()
        for n,p in model.named_parameters():
            if p.grad is not None: F[n]+=p.grad.detach()**2
        c+=1
        if c*16>=nmax: break
    for n in F: F[n]/=max(c,1)
    return F
import csv

# ===== (1) JOINT DEPLOYMENT MODEL =====
print("\n===== JOINT deployment model (all conditions + augmentation) =====",flush=True)
joint=fresh(); joint=train(joint,balance_idx(where(split="train")),epochs=12,aug=True)
cond=[]
for m in ORDER:
    for pl in PLANES[m]:
        for R in RSET:
            a=eval_cond(joint,m,pl,R); cond.append((m,pl,R,a))
            print(f"  {NM[m][0]} {pl:8s} R{R}: AUC {a:.3f}",flush=True)
overall=auc_on(joint,np.concatenate([where(model="real",split="test"),where(split="test",label=1)]))
print(f"  OVERALL held-out AUC (any input): {overall:.3f}",flush=True)
with open(f"{OUT}/joint_per_condition.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["model","plane","R","auc"]); [w.writerow([m,pl,R,round(a,4)]) for m,pl,R,a in cond]
    w.writerow(["OVERALL","","",round(overall,4)])
torch.save(joint.state_dict(),f"{OUT}/meta_joint_deploy.pt"); print("saved meta_joint_deploy.pt",flush=True)

# ===== (2) CONTINUAL LEARNING vs NAIVE =====
def run_cl(use_cl):
    model=fresh(); replay=[]; tasks=[]; lam=(4000.0 if use_cl else 0.0); rows=[]
    for k in range(len(ORDER)):
        pos=where(model=ORDER[k],split="train"); neg=where(model="real",split="train")
        if len(neg)>len(pos): neg=rng.choice(neg,len(pos),replace=False)
        idx=np.concatenate([pos,neg])
        train(model,idx,epochs=8,aug=True,replay=(replay if use_cl else None),tasks=(tasks if use_cl else None),lam=lam)
        for j,m in enumerate(ORDER):
            # held-out per model, aggregated over its planes/R
            negs=where(model="real",split="test"); poss=where(model=m,split="test")
            rows.append((k,m,auc_on(model,np.concatenate([negs,poss])),j<=k))
        if use_cl:
            tasks.append(({n:p.detach().clone() for n,p in model.named_parameters()},fisher(model,idx)))
            samp=list(rng.choice(idx,min(200,len(idx)),replace=False)); replay+=[(int(i),int(LAB[i])) for i in samp]
        print(f"[{'CL' if use_cl else 'naive'}] stage {k+1}: "+" ".join(f"{NM[m][0]}={[r[2] for r in rows if r[0]==k and r[1]==m][0]:.3f}" for m in ORDER),flush=True)
    return rows
print("\n===== CONTINUAL LEARNING (replay+EWC) =====",flush=True); cl_rows=run_cl(True)
print("\n===== NAIVE baseline =====",flush=True); naive_rows=run_cl(False)

# ===== (3) LEAVE-ONE-MODEL-OUT =====
print("\n===== LEAVE-ONE-MODEL-OUT (generalization to a fully unseen architecture) =====",flush=True)
loo=[]
for h in ORDER:
    pos=np.concatenate([where(model=o,split="train") for o in ORDER if o!=h]); neg=where(model="real",split="train")
    if len(neg)>len(pos): neg=rng.choice(neg,len(pos),replace=False)
    m=train(fresh(),np.concatenate([pos,neg]),epochs=10,aug=True)
    a=auc_on(m,np.concatenate([where(model="real",split="test"),where(model=h,split="test")]))
    loo.append((h,a)); print(f"  held out {NM[h][0]} (trained on the other two): AUC {a:.3f}",flush=True)

# ---- save tables + figure ----
with open(f"{OUT}/meta_v2_results.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["experiment","method","stage_or_heldout","eval_model","auc","seen"])
    for k,m,a,seen in cl_rows: w.writerow(["continual","CL",k+1,m,round(a,4),int(seen)])
    for k,m,a,seen in naive_rows: w.writerow(["continual","naive",k+1,m,round(a,4),int(seen)])
    for h,a in loo: w.writerow(["leave_one_out","LOO",h,h,round(a,4),0])
print("wrote meta_v2_results.csv",flush=True)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
col={"dcnn":"#142A4A","zssr":"#1E7A33","ppn":"#C0392B"}
fig,ax=plt.subplots(1,2,figsize=(13,5))
for m in ORDER:
    pts=[(k+1,a) for (k,mm,a,s) in cl_rows if mm==m]; xs,ys=zip(*pts)
    ax[0].plot(xs,ys,"o-",color=col[m],lw=2,ms=7,label=NM[m])
ax[0].set_xticks([1,2,3]); ax[0].set_xticklabels(["A","A+B","A+B+C"]); ax[0].set_ylim(0.45,1.02); ax[0].axhline(0.5,ls="--",c="gray",lw=1)
ax[0].set_xlabel("Continual-learning stage"); ax[0].set_ylabel("Held-out detection AUC"); ax[0].set_title("Meta-classifier generalization"); ax[0].legend(); ax[0].grid(alpha=0.3)
for tag,rows,c,ls in [("CL",cl_rows,"#1E7A33","o-"),("naive",naive_rows,"#C0392B","s--")]:
    pts=[(k+1,a) for (k,mm,a,s) in rows if mm=="dcnn"]; xs,ys=zip(*pts); ax[1].plot(xs,ys,ls,color=c,lw=2,ms=7,label=tag)
ax[1].set_xticks([1,2,3]); ax[1].set_xticklabels(["A","A+B","A+B+C"]); ax[1].set_ylim(0.45,1.02)
ax[1].set_xlabel("Stage"); ax[1].set_ylabel("AUC on Model A"); ax[1].set_title("Anti-forgetting (CL vs naive)"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_meta_v2.png",dpi=140); print("wrote fig5_meta_v2.png",flush=True)
print("\n==== meta v2 training done ====",flush=True)
print(f"DEPLOYMENT MODEL overall held-out AUC (any input): {overall:.3f}",flush=True)
