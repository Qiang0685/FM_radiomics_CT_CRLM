#!/usr/bin/env python3
"""Nested linear-probe TotalFM sensitivity across three liver masks."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEEDS=[42,0,1,7,2024]; C_GRID=[.01,.1,1,10]
TASKS=[('Steatosis','steatosis'),('NAS >= 3','nas3'),('Sinusoidal dilatation','sinusoidal_dilatation')]
ROIS=['TotalSegmentator liver','TCIA liver','Resected tumor-free liver']

def read_matrix(path,ids):
 d=pd.read_csv(path,dtype={'case_id':str});keys=d.case_id.astype(str).str.extract(r'(\d{4})',expand=False)
 if keys.isna().any() or keys.duplicated().any():raise ValueError(f'Bad IDs: {path}')
 d.index=keys;x=d.loc[ids].drop(columns=['case_id']).to_numpy(float)
 if x.shape!=(len(ids),768) or not np.isfinite(x).all():raise ValueError(f'Bad TotalFM matrix {path}: {x.shape}')
 return x
def midrank(x):
 ix=np.argsort(x);z=x[ix];r=np.zeros(len(x));i=0
 while i<len(x):
  j=i+1
  while j<len(x) and z[j]==z[i]:j+=1
  r[i:j]=(i+j-1)/2+1;i=j
 out=np.empty(len(x));out[ix]=r;return out
def delong(y,a,b):
 ix=np.argsort(-y);yy=y[ix];m=int(yy.sum());n=len(yy)-m;vec=[]
 for s in [a[ix],b[ix]]:
  tx,ty,tz=midrank(s[:m]),midrank(s[m:]),midrank(s);vec.append(((tz[:m]-tx)/n,1-(tz[m:]-ty)/m))
 cov=np.atleast_2d(np.cov(np.vstack([v[0] for v in vec])))/m;cov+=np.atleast_2d(np.cov(np.vstack([v[1] for v in vec])))/n
 delta=roc_auc_score(y,b)-roc_auc_score(y,a);v=float(cov[0,0]+cov[1,1]-2*cov[0,1])
 if v<=1e-15 and abs(delta)<=1e-12:return delta,1.0
 return delta,float(2*norm.sf(abs(delta/np.sqrt(v)))) if v>1e-15 else np.nan
def holm(p):
 p=np.asarray(p);ix=np.argsort(p);q=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[ix]).clip(0,1);o=np.empty(len(p));o[ix]=q;return o
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--tcia-embedding',type=Path,required=True);ap.add_argument('--oof',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--bootstrap',type=int,default=1000);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 matrix_paths=[a.data/'03_TotalFM_liver_TS_197x768.csv',a.tcia_embedding,a.data/'03b_totalfm_resected_liver_197x768.csv']
 all_summary=[];pairrows=[];pred_out=[]
 with threadpool_limits(limits=1):
  for ti,(task,slug) in enumerate(TASKS):
   oof_path=a.oof/f'oof_{slug}.csv'
   if not oof_path.exists(): oof_path=a.oof/f'main_oof_{slug}.csv'
   labels=pd.read_csv(oof_path,dtype={'case_id':str});ids=labels.case_id.astype(str).str.extract(r'(\d{4})',expand=False).tolist();y=labels.y_true.to_numpy(int)
   roi_probs=[]
   for ri,(roi,path) in enumerate(zip(ROIS,matrix_paths)):
    X=read_matrix(path,ids); seed_probs=[];seed_aucs=[]
    for seed in SEEDS:
     p=np.full(len(y),np.nan);outer=StratifiedKFold(5,shuffle=True,random_state=seed)
     for fold,(tr,te) in enumerate(outer.split(X,y),1):
      inner=StratifiedKFold(3,shuffle=True,random_state=seed*100+fold)
      pipe=make_pipeline(StandardScaler(),LogisticRegression(penalty='l2',solver='lbfgs',max_iter=5000,tol=1e-4))
      search=GridSearchCV(pipe,{'logisticregression__C':C_GRID},scoring='roc_auc',cv=inner,n_jobs=1,refit=True)
      search.fit(X[tr],y[tr]);p[te]=search.predict_proba(X[te])[:,1]
     if not np.isfinite(p).all():raise RuntimeError(f'Incomplete OOF: {task}/{roi}/{seed}')
     seed_probs.append(p);seed_aucs.append(roc_auc_score(y,p))
    avg=np.mean(seed_probs,axis=0);roi_probs.append(avg)
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);boot=np.empty(a.bootstrap);rng=np.random.default_rng(4200+ti)
    for b in range(a.bootstrap):
     ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)];boot[b]=roc_auc_score(y[ix],avg[ix])
    lo,hi=np.quantile(boot,[.025,.975]);all_summary.append({'Task':task,'ROI':roi,'N':len(y),'Events':int(y.sum()),
      'Mean_seed_AUC':float(np.mean(seed_aucs)),'SD_seed_AUC':float(np.std(seed_aucs,ddof=1)),
      'AUC_seed_averaged_OOF':roc_auc_score(y,avg),'AUC_CI_low':lo,'AUC_CI_high':hi})
    pred_out.append(pd.DataFrame({'case_id':labels.case_id,'y_true':y,'Task':task,'ROI':roi,'OOF_probability':avg}))
   start=len(pairrows)
   for i in range(3):
    for j in range(i+1,3):
     delta,p=delong(y,roi_probs[i],roi_probs[j]);pairrows.append({'Task':task,'Baseline_ROI':ROIS[i],'Comparator_ROI':ROIS[j],
      'Signed_delta_AUC_comparator_minus_baseline':delta,'DeLong_p':p})
   adj=holm([r['DeLong_p'] for r in pairrows[start:]])
   for r,q in zip(pairrows[start:],adj):r['Holm_p_within_task_three_ROI_pairs']=q
 summary=pd.DataFrame(all_summary);summary.to_csv(a.out/'totalfm_roi_performance_raw.csv',index=False)
 pairs=pd.DataFrame(pairrows);pairs.to_csv(a.out/'totalfm_roi_pairwise_delong_raw.csv',index=False)
if __name__=='__main__':main()
