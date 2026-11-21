"""Why hip rotation spread grows on zoomed re-pose: hip-angle jumps, hip width, and the effect of masking collapsed hips.

Usage: check_hip_spread.py <dataset features npz> <crop features npz>

Compares the two files on the strokes they share. Hip width is the distance between the hip keypoints in torso
lengths; when it is tiny the hip line angle is noise. Prints spread (std over strokes, contact +-3 frames) of
hip_rotation and separation with frames masked below several hip-width thresholds.
"""
import sys
import numpy as np
a=np.load(sys.argv[1]); b=np.load(sys.argv[2])
F=list(a["feature_names"]); H=F.index("hip_rotation"); S=F.index("shoulder_rotation")
meta={m[0]:m for m in a["meta"]}
jumps={"ds":[], "crop":[]}; diff=[]; atc={"ds":[], "crop":[]}; hipw={"ds":[],"crop":[]}
for m in b["meta"]:
    sid=m[0]; c=int(m[6])-int(m[8])
    for k,d in (("ds",a),("crop",b)):
        h=d[f"{sid}/feat"][:,H]; dh=np.abs(np.diff(h)); jumps[k].append(np.nansum(dh>60))
        atc[k].append(h[c] if c<len(h) else np.nan)
        kp=d[f"{sid}/kp"]; w=np.linalg.norm(kp[:,11,:2]-kp[:,12,:2],axis=1); t=np.linalg.norm((kp[:,5,:2]+kp[:,6,:2])/2-(kp[:,11,:2]+kp[:,12,:2])/2,axis=1)
        hipw[k].append(np.nanmedian(w/t))
    diff.append(np.nanmedian(np.abs(b[f"{sid}/feat"][:,H]-a[f"{sid}/feat"][:,H])))
for k in jumps:
    j=np.array(jumps[k]); print(k,"strokes with hip jumps>60deg/frame:",(j>0).sum(),"of",len(j),"| hip rot at contact: std",round(np.nanstd(atc[k]),1),"IQR",np.nanpercentile(atc[k],[25,75]).round(1),"| hip width/torso median",round(np.nanmedian(hipw[k]),3))
print("median |crop-ds| hip rot per stroke pct",np.nanpercentile(diff,[25,50,75,90]).round(1))
print()
for k,d in (("ds",a),("crop",b)):
    mags=[]; runs=[]; hipdx=[]
    for m in b["meta"]:
        sid=m[0]; h=d[f"{sid}/feat"][:,H]; dh=np.diff(h); big=np.abs(dh)>60
        mags+=list(np.abs(dh[big]))
        kp=d[f"{sid}/kp"]; hipdx+=list(((kp[:,12,0]-kp[:,11,0])/np.linalg.norm((kp[:,5,:2]+kp[:,6,:2])/2-(kp[:,11,:2]+kp[:,12,:2])/2,axis=1))[big.nonzero()[0]])
    mags=np.array(mags); print(k,"big jumps",len(mags),"size pct",np.percentile(mags,[10,50,90]).round(0) if len(mags) else "-", "| hip dx/torso at jump pct", np.nanpercentile(np.abs(hipdx),[10,50,90]).round(2) if hipdx else "-")
# overall share of frames where hip dx (horizontal hip separation) is tiny
for k,d in (("ds",a),("crop",b)):
    v=[]
    for m in b["meta"]:
        kp=d[f"{m[0]}/kp"]; t=np.linalg.norm((kp[:,5,:2]+kp[:,6,:2])/2-(kp[:,11,:2]+kp[:,12,:2])/2,axis=1)
        v+=list(np.linalg.norm(kp[:,12,:2]-kp[:,11,:2],axis=1)/t)
    v=np.array(v); print(k,"frames with hip width < 0.15 torso:",round(100*np.nanmean(v<0.15),1),"%")
print()
SEP=F.index("separation")
for thr in (0, 0.1, 0.15, 0.2):
    out=[]
    for k,d in (("ds",a),("crop",b)):
        for col in (H, SEP):
            vals=[]
            for m in b["meta"]:
                sid=m[0]; c=int(m[6])-int(m[8]); kp=d[f"{sid}/kp"]; f=d[f"{sid}/feat"][:,col].copy()
                t=np.linalg.norm((kp[:,5,:2]+kp[:,6,:2])/2-(kp[:,11,:2]+kp[:,12,:2])/2,axis=1)
                w=np.linalg.norm(kp[:,12,:2]-kp[:,11,:2],axis=1)/t
                f[w<thr]=np.nan
                vals+=list(f[max(c-3,0):c+4])
            vals=np.array(vals); out.append(f"{k} {F[col][:4]} std {np.nanstd(vals):.1f} miss {100*np.isnan(vals).mean():.0f}%")
    print(f"hip width >= {thr}: ", " | ".join(out))
