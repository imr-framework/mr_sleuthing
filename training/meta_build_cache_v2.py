# Comprehensive meta-classifier cache: real + per-model recon slices across MODELS x R{4,8} x planes{axial,sagittal}.
# One volume read -> many slices (axial + sagittal, multiple positions). Global subject-level split (no leakage).
# ppn is per-slice (axial only). Volume models (dcnn,zssr) contribute both planes -> gives ZSSR its through-plane view.
import os, glob, time, warnings, numpy as np, nibabel as nib
warnings.filterwarnings("ignore")
from scipy.ndimage import zoom
from collections import defaultdict
DP=os.environ.get("MRS_DATAPOOL","data/jhu26_datapool")  # folder with pool/ (originals) and recons/{dcnn,zssr,ppn}/
POOL=f"{DP}/pool"; REC=f"{DP}/recons"
OUT=os.environ.get("MRS_OUT","outputs/meta_classifier"); os.makedirs(OUT,exist_ok=True)
CACHE=f"{OUT}/meta_slices_v2.npz"
SIZE=320; MODELS=["dcnn","zssr","ppn"]; RS=[4,8]; MAXPER=32; NAX=4; NSAG=4; NPPN=4
def n01(s): s=np.abs(s).astype(np.float32); lo,hi=np.percentile(s,1),np.percentile(s,99); return np.clip((s-lo)/max(hi-lo,1e-6),0,1)
def ras_T(a): o=nib.orientations.io_orientation(a); return nib.orientations.ornt_transform(o,nib.orientations.axcodes2ornt(("R","A","S")))
def to_ras(a,af): return nib.orientations.apply_orientation(a,ras_T(af))
def rad_axial(r,z): return np.fliplr(np.flipud(r[:,:,z].T))
def rad_sag(r,x): return np.flipud(r[x,:,:].T)
def central(n,k): c=n//2; h=k//2; return list(range(max(0,c-h),min(n,max(0,c-h)+k)))
def to320(sl): return zoom(n01(sl),(SIZE/sl.shape[0],SIZE/sl.shape[1]),order=1)
def split_of(subj): return "test" if (sum(ord(c) for c in subj)%10)<3 else "train"
_pc={}
def pool_ras(subj):
    if subj in _pc: return _pc[subj]
    p=f"{POOL}/{subj}.nii.gz"
    if not os.path.exists(p): _pc[subj]=None; return None
    for _ in range(3):
        try:
            v=nib.load(p); a=np.asarray(v.dataobj,dtype=np.float32); a=a[...,0] if a.ndim==4 else a; _pc[subj]=to_ras(a,v.affine); return _pc[subj]
        except Exception: time.sleep(2)
    _pc[subj]=None; return None
def vol_ras(path):
    for _ in range(3):
        try:
            v=nib.load(path); a=np.asarray(v.dataobj,dtype=np.float32); a=a[...,0] if a.ndim==4 else a; return to_ras(a,v.affine)
        except Exception: time.sleep(2)
    return None
def rds(model,R):
    if model=="dcnn": return (f"{REC}/dcnn" if R==4 else f"{REC}/dcnn_sweep/R{R}"), f"_dcnnR{R}.nii.gz"
    if model=="zssr": return (f"{REC}/zssr" if R==4 else f"{REC}/zssr_sweep/R{R}"), f"_kspace_x{R}_zssr.nii.gz"
    return (f"{REC}/ppn" if R==4 else f"{REC}/ppn_sweep/R{R}"), f"__ppnR{R}.nii.gz"
def capped(model,R):
    d,suf=rds(model,R); files=sorted(glob.glob(f"{d}/*{suf}"))
    sf=defaultdict(list); sm={}
    for f in files:
        b=os.path.basename(f); s=(b.split("__z")[0] if model=="ppn" else b.replace(suf,"")); sf[s].append(f); sm[s]=s.split("__")[-1]
    bym=defaultdict(list)
    for s in sorted(sf): bym[sm[s]].append(s)
    keep=set(s for m in ["T1","T2","FLAIR"] for s in bym[m][:MAXPER])
    return {s:sf[s] for s in keep},suf

imgs=[];lab=[];mod=[];subj=[];spl=[];plane=[];Rtag=[]
real_seen=set()
def add(img,l,m,s,sp,pl,R):
    imgs.append(img.astype(np.float16)); lab.append(l); mod.append(m); subj.append(s); spl.append(sp); plane.append(pl); Rtag.append(R)
for R in RS:
    for model in MODELS:
        keep,suf=capped(model,R); print(f"[{model} R{R}] {len(keep)} subjects",flush=True); n0=len(imgs)
        for s,fl in keep.items():
            sp=split_of(s)
            if model=="ppn":
                ras=pool_ras(s)
                if ras is None: continue
                cnt=0
                for rp in fl:
                    if cnt>=NPPN: break
                    b=os.path.basename(rp)
                    try: z=int(b.split("__z")[1].split("__")[0])
                    except: continue
                    if z>=ras.shape[2]: continue
                    try: r=n01(np.asarray(nib.load(rp).dataobj,dtype=np.float32).squeeze())
                    except Exception: continue
                    o=to320(rad_axial(ras,z))
                    if r.shape!=(SIZE,SIZE): r=zoom(n01(r),(SIZE/r.shape[0],SIZE/r.shape[1]),order=1)
                    m=(o>0.03).astype(np.float32)
                    add(n01(r*m),1,model,s,sp,"axial",R); cnt+=1
                    rk=f"{s}__axial__{z}"
                    if rk not in real_seen: real_seen.add(rk); add(n01(o*m),0,"real",s,sp,"axial",R)
            else:
                ras=pool_ras(s); rr=vol_ras(fl[0])
                if ras is None or rr is None or rr.shape!=ras.shape: continue
                for z in central(ras.shape[2],NAX):
                    o=to320(rad_axial(ras,z)); r=to320(rad_axial(rr,z)); m=(o>0.03).astype(np.float32)
                    add(n01(r*m),1,model,s,sp,"axial",R)
                    rk=f"{s}__axial__{z}"
                    if rk not in real_seen: real_seen.add(rk); add(n01(o*m),0,"real",s,sp,"axial",R)
                for x in central(ras.shape[0],NSAG):
                    o=to320(rad_sag(ras,x)); r=to320(rad_sag(rr,x)); m=(o>0.03).astype(np.float32)
                    add(n01(r*m),1,model,s,sp,"sagittal",R)
                    rk=f"{s}__sagittal__{x}"
                    if rk not in real_seen: real_seen.add(rk); add(n01(o*m),0,"real",s,sp,"sagittal",R)
        print(f"  [{model} R{R}] +{len(imgs)-n0} slices (total {len(imgs)})",flush=True)
imgs=np.stack(imgs)
np.savez_compressed(CACHE, imgs=imgs, labels=np.array(lab,np.int8), model=np.array(mod), subj=np.array(subj),
                    split=np.array(spl), plane=np.array(plane), R=np.array(Rtag,np.int16))
print("\n=== summary ===",flush=True)
lab=np.array(lab); mod=np.array(mod); spl=np.array(spl); plane=np.array(plane)
print("total",len(lab),"| reals",int((lab==0).sum()),"recons",int((lab==1).sum()),flush=True)
for m in MODELS+["real"]:
    for pl in ["axial","sagittal"]:
        sel=(mod==m)&(plane==pl)
        if sel.sum(): print(f"  {m:5s} {pl:8s}: train {int((sel&(spl=='train')).sum()):4d}  test {int((sel&(spl=='test')).sum()):4d}",flush=True)
print("wrote",CACHE,f"({os.path.getsize(CACHE)/1e6:.0f} MB)",flush=True)
