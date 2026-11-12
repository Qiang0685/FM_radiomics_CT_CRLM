#!/usr/bin/env python3
"""Fresh three-ROI Radiomics sensitivity analysis for the reproducibility run.

This script re-fits the same frozen linear-probe workflow using only the three
current Radiomics matrices in ``Data``.  It does not read any prior result
table; the plotting geometry remains in ``08B_plot_roi_sensitivity.py``.
"""
from __future__ import annotations
import argparse, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

TASKS = [("Steatosis", "steatosis"), ("NAS >= 3", "nas3"), ("Sinusoidal dilatation", "sinusoidal_dilatation")]
SETS = {
    "whole_noVes": "06c_radiomics_wholeLiver_197x1130.csv",
    "whole_noVesTumor": "06b_radiomics_wholeLiver_tumorFree_197x1130.csv",
    "resected_noVesTumor": "06_radiomics_resected_TumorFree_197x1130.csv",
}

def load_probe(path: Path):
    spec = importlib.util.spec_from_file_location("probe", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def bootstrap_auc(y, p, n=1000, seed=4400):
    rng=np.random.default_rng(seed); pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    d=np.empty(n)
    for i in range(n):
        ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        d[i]=roc_auc_score(y[ix],p[ix])
    return float(roc_auc_score(y,p)), np.quantile(d,[.025,.975])

def midrank(x):
    ix=np.argsort(x); z=x[ix]; r=np.zeros(len(x)); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and z[j]==z[i]: j+=1
        r[i:j]=(i+j-1)/2+1; i=j
    out=np.empty(len(x)); out[ix]=r; return out

def delong(y,a,b):
    ix=np.argsort(-y); yy=y[ix]; m=int(yy.sum()); n=len(yy)-m; vv=[]
    for s in (a[ix],b[ix]):
        tx,ty,tz=midrank(s[:m]),midrank(s[m:]),midrank(s)
        vv.append(((tz[:m]-tx)/n, 1-(tz[m:]-ty)/m))
    cov=np.atleast_2d(np.cov(np.vstack([v[0] for v in vv])))/m
    cov+=np.atleast_2d(np.cov(np.vstack([v[1] for v in vv])))/n
    delta=roc_auc_score(y,b)-roc_auc_score(y,a); v=float(cov[0,0]+cov[1,1]-2*cov[0,1])
    return delta, float(2*norm.sf(abs(delta/np.sqrt(v)))) if v>0 else np.nan

def holm(p):
    p=np.asarray(p,float); ix=np.argsort(p); q=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[ix]).clip(0,1); out=np.empty(len(p)); out[ix]=q; return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--code-root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--bootstrap',type=int,default=1000); a=ap.parse_args()
    out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    probe=load_probe(a.code_root/'01A_unified_linear_probe.py')
    ids, labels, _, _ = probe.load_inputs(a.data_root)
    performances=[]; scores_by_task={task:{} for task,_ in TASKS}
    for set_name, file_name in SETS.items():
        frame=pd.read_csv(a.data_root/file_name,dtype={'case_id':str}); frame.index=frame.case_id.str.extract(r'(\d{4})',expand=False)
        X=frame.drop(columns='case_id').loc[ids].to_numpy(float)
        if X.shape[1] != 1130 or not np.isfinite(X).all(): raise ValueError(f'Invalid matrix: {file_name}')
        for ti,(task,slug) in enumerate(TASKS):
            y=labels[task]; p,_,seed_probs=probe.fit_outer_oof(X,y,task,'Radiomics',probe.SEEDS,inner_splits=3,return_seed_probs=True)
            scores_by_task[task][set_name]=p
            # Match the primary analysis bootstrap exactly: task-specific
            # seed 4200 + task index, with 1000 class-stratified resamples.
            auc,ci=bootstrap_auc(y,p,a.bootstrap,4200+ti)
            seed_aucs=[roc_auc_score(y,s) for s in seed_probs]
            performances.append({'Radiomics_set':set_name,'Task':task,'N':len(y),'Positive':int(y.sum()),'Features':X.shape[1],
                                 'Mean_seed_AUC':float(np.mean(seed_aucs)),'SD_seed_AUC':float(np.std(seed_aucs,ddof=1)),
                                 'OOF_AUC':auc,'AUC_CI_low':ci[0],'AUC_CI_high':ci[1]})
    perf=pd.DataFrame(performances); perf.to_csv(out/'radiomics_roi_performance_raw.csv',index=False,float_format='%.17g')
    pairs=[]
    for ti,(task,slug) in enumerate(TASKS):
        y=labels[task]; fs=scores_by_task[task]; rows=[]
        names=list(SETS)
        for i in range(3):
            for j in range(i+1,3):
                d,p=delong(y,fs[names[i]],fs[names[j]]); rows.append({'Task':task,'ROI_1':names[i],'ROI_2':names[j],'Signed_delta_AUC_ROI2_minus_ROI1':d,'DeLong_p':p})
        adj=holm([r['DeLong_p'] for r in rows])
        for r,q in zip(rows,adj): r['Holm_p_within_task']=q
        pairs.extend(rows)
    pair=pd.DataFrame(pairs); pair.to_csv(out/'radiomics_roi_pairwise_delong_raw.csv',index=False,float_format='%.17g')
    print(f'Saved fresh Radiomics ROI analysis to {out}')

if __name__=='__main__': main()
