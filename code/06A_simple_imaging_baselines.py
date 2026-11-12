#!/usr/bin/env python3
"""Task-specific simple imaging baseline AUC and paired DeLong comparisons."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
MODELS=['Merlin','RADAR','TotalFM','Jolia','OrganLens','Radiomics']
TASKS=[('Steatosis','steatosis'),('Sinusoidal dilatation','sinusoidal_dilatation')]
def mr(x):
 ix=np.argsort(x);z=x[ix];r=np.zeros(len(x));i=0
 while i<len(x):
  j=i+1
  while j<len(x) and z[j]==z[i]:j+=1
  r[i:j]=(i+j-1)/2+1;i=j
 out=np.empty(len(x));out[ix]=r;return out
def dtest(y,a,b):
 ix=np.argsort(-y);yy=y[ix];m=int(yy.sum());n=len(yy)-m;vv=[]
 for s in [a[ix],b[ix]]:
  tx,ty,tz=mr(s[:m]),mr(s[m:]),mr(s);vv.append(((tz[:m]-tx)/n,1-(tz[m:]-ty)/m))
 c=np.atleast_2d(np.cov(np.vstack([v[0] for v in vv])))/m;c+=np.atleast_2d(np.cov(np.vstack([v[1] for v in vv])))/n
 delta=roc_auc_score(y,b)-roc_auc_score(y,a);v=float(c[0,0]+c[1,1]-2*c[0,1]);return delta,float(2*norm.sf(abs(delta/np.sqrt(v)))) if v>0 else np.nan
def holm(p):
 p=np.asarray(p);ix=np.argsort(p);q=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[ix]).clip(0,1);z=np.empty(len(p));z[ix]=q;return z
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--oof',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--bootstrap',type=int,default=1000);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 inp=pd.read_csv(a.inputs,encoding='latin1',dtype={'case_id':str});inp['case_key']=inp.case_id.astype(str).str.extract(r'(\d{4})',expand=False)
 if 'mean_spleen_hu' in inp.columns:
  inp['mean_spleen_hu']=inp['mean_spleen_hu'].fillna(inp['mean_spleen_hu'].median())
 inp['spleen_volume_ml']=inp['spleen_volume_ml'].fillna(inp['spleen_volume_ml'].median())
 baselines={'Mean liver parenchyma HU':('lower',-inp.mean_liver_resected_hu.to_numpy()),'Spleen volume':('higher',inp.spleen_volume_ml.to_numpy())}
 imap=inp.set_index('case_key');tasks={'Steatosis':['Mean liver parenchyma HU'],'Sinusoidal dilatation':['Spleen volume']};rows=[];auc_ci=[]
 task_seed_index={'Steatosis':0,'NAS >= 3':1,'Sinusoidal dilatation':2}
 for task,slug in TASKS:
  # Match the primary analysis bootstrap stream exactly (4200 + task index).
  rng=np.random.default_rng(4200+task_seed_index[task])
  path=a.oof/f'oof_{slug}.csv'
  if not path.exists(): path=a.oof/f'main_oof_{slug}.csv'
  d=pd.read_csv(path);y=d.y_true.to_numpy(int)
  for name in tasks[task]:
   direction,_=baselines[name];keys=d.case_id.astype(str).str.extract(r'(\d{4})',expand=False).tolist()
   raw={'Mean liver parenchyma HU':imap.mean_liver_resected_hu,'Spleen volume':imap.spleen_volume_ml}[name]
   v=np.asarray([raw.loc[k] for k in keys],float);score=-v if direction=='lower' else v
   base_auc=roc_auc_score(y,score);ps=[];tmp=[]
   pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);base_draw=np.empty(a.bootstrap);model_draw={m:np.empty(a.bootstrap) for m in MODELS}
   for b in range(a.bootstrap):
    ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
    base_draw[b]=roc_auc_score(y[ix],score[ix])
    for m in MODELS:model_draw[m][b]=roc_auc_score(y[ix],d[m].to_numpy(float)[ix])
   lo,hi=np.quantile(base_draw,[.025,.975]);auc_ci.append({'Task':task,'Predictor':name,'N':len(y),'Events':int(y.sum()),'AUC':base_auc,'CI_low':lo,'CI_high':hi,'Role':'Simple baseline'})
   for model in MODELS:
    delta,p=dtest(y,score,d[model].to_numpy(float));tmp.append({'Task':task,'Baseline':name,'Direction':direction,'Comparator':model,'N':len(y),'Events':int(y.sum()),
      'Baseline_AUC':base_auc,'Comparator_AUC':roc_auc_score(y,d[model]),'Signed_delta_AUC_model_minus_baseline':delta,'DeLong_p':p})
    lo,hi=np.quantile(model_draw[model],[.025,.975]);auc_ci.append({'Task':task,'Predictor':model,'N':len(y),'Events':int(y.sum()),'AUC':roc_auc_score(y,d[model]),'CI_low':lo,'CI_high':hi,'Role':'OOF model comparator'})
    ps.append(p)
   adj=holm(ps)
   for r,p in zip(tmp,adj):r['Holm_p_within_baseline_task_six_models']=p
   rows.extend(tmp)
 pd.DataFrame(rows).to_csv(a.out/'simple_baseline_vs_models_delong.csv',index=False)
 pd.DataFrame(auc_ci).to_csv(a.out/'simple_baseline_comparison_auc_ci.csv',index=False)
if __name__=='__main__':main()
