#!/usr/bin/env python3
"""Primary mean seed-level AUC plot/table and ordered signed DeLong contrasts."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

MODELS=['Merlin','RADAR','TotalFM','Jolia','OrganLens','Radiomics']
TASKS=[('Steatosis','steatosis'),('NAS >= 3','nas3'),('Sinusoidal dilatation','sinusoidal_dilatation')]
TASK_LABELS={'NAS >= 3':'NAS ≥ 3'}
COLORS=['#4C78A8','#59A14F','#F28E2B','#B279A2','#76B7B2','#7F7F7F']

def midrank(x):
 ix=np.argsort(x); z=x[ix]; r=np.zeros(len(x)); i=0
 while i<len(x):
  j=i+1
  while j<len(x) and z[j]==z[i]: j+=1
  r[i:j]=(i+j-1)/2+1; i=j
 out=np.empty(len(x)); out[ix]=r; return out
def delong(y,a,b):
 ix=np.argsort(-y); yy=y[ix]; m=int(yy.sum()); n=len(yy)-m; vv=[]
 for s in [a[ix],b[ix]]:
  tx,ty,tz=midrank(s[:m]),midrank(s[m:]),midrank(s)
  vv.append(((tz[:m]-tx)/n,1-(tz[m:]-ty)/m))
 cov=np.atleast_2d(np.cov(np.vstack([v[0] for v in vv])))/m
 cov+=np.atleast_2d(np.cov(np.vstack([v[1] for v in vv])))/n
 delta=roc_auc_score(y,b)-roc_auc_score(y,a); v=float(cov[0,0]+cov[1,1]-2*cov[0,1])
 return delta,float(2*norm.sf(abs(delta/np.sqrt(v)))) if v>0 else np.nan
def holm(p):
 p=np.asarray(p); ix=np.argsort(p); q=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[ix]).clip(0,1); out=np.empty(len(p)); out[ix]=q; return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--oof',type=Path,required=True); ap.add_argument('--seed-performance',type=Path,required=True); ap.add_argument('--out',type=Path,required=True,help='Analysis-table output directory.'); ap.add_argument('--figure-dir',type=Path,required=True,help='Figure-only output directory (PDF and PNG only).'); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); a.figure_dir.mkdir(parents=True,exist_ok=True)
 s=pd.read_csv(a.seed_performance); rows=[]
 for task,_ in TASKS:
  d=s[s.Task==task]
  for model in MODELS:
   q=d[d.Model==model]
   rows.append({'Task':task,'Model':model,'N':int(q.N.iloc[0]),'Events':int(q.Events.iloc[0]),
                'Mean_seed_AUC':q.AUC_raw_seed_OOF.mean(),'SD_seed_AUC':q.AUC_raw_seed_OOF.std(ddof=1)})
 primary=pd.DataFrame(rows); primary.to_csv(a.out/'primary_mean_seed_level_auc.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(180/25.4,68/25.4),sharey=True)
 for ax,(task,_) in zip(axes,TASKS):
  d=primary[primary.Task==task].set_index('Model').loc[MODELS]; x=np.arange(6)
  ax.bar(x,d.Mean_seed_AUC,color=COLORS,width=.68,yerr=d.SD_seed_AUC,error_kw={'ecolor':'#30343B','elinewidth':.9,'capsize':2,'capthick':.9})
  ax.axhline(.5,color='#777777',linestyle='--',linewidth=.8); ax.set_title(TASK_LABELS.get(task,task),fontsize=7.5)
  ax.set_xticks(x,MODELS,rotation=48,ha='right',fontsize=5.5); ax.grid(axis='y',color='#E1E4E8',linewidth=.5);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('OOF AUC');axes[0].set_ylim(0,1.02);axes[0].set_yticks(np.arange(0,1.01,.2))
 # Leave sufficient physical margin for the full vertical y-axis label.
 fig.subplots_adjust(left=.115,right=.995,bottom=.31,top=.91,wspace=.18)
 fig.savefig(a.figure_dir/'Figure_2_mean_seed_level_discrimination.pdf',facecolor='white');fig.savefig(a.figure_dir/'Figure_2_mean_seed_level_discrimination.png',dpi=400,facecolor='white');plt.close(fig)
 out=[]
 for task,slug in TASKS:
  oof_path=a.oof/f'oof_{slug}.csv'
  if not oof_path.exists(): oof_path=a.oof/f'main_oof_{slug}.csv'
  d=pd.read_csv(oof_path); y=d.y_true.to_numpy(int); rad=d.Radiomics.to_numpy(float)
  start=len(out)
  # FM minus Radiomics; reviewer-specified five-comparison Holm family.
  for fm in MODELS[:-1]:
   delta,p=delong(y,rad,d[fm].to_numpy(float));out.append({'Task':task,'Family':'FM_vs_Radiomics','Baseline':'Radiomics','Comparator':fm,'Signed_delta_AUC':delta,'DeLong_p':p})
  adj=holm([r['DeLong_p'] for r in out[start:start+5]])
  for r,v in zip(out[start:start+5],adj):r['Holm_p_within_5_FM_vs_Radiomics']=v
  start=len(out)
  for i,base in enumerate(MODELS[:-1]):
   for comp in MODELS[i+1:-1]:
    delta,p=delong(y,d[base].to_numpy(float),d[comp].to_numpy(float));out.append({'Task':task,'Family':'FM_vs_FM_exploratory','Baseline':base,'Comparator':comp,'Signed_delta_AUC':delta,'DeLong_p':p})
  adj=holm([r['DeLong_p'] for r in out[start:]])
  for r,v in zip(out[start:],adj):r['Holm_p_within_10_FM_pairs']=v
 table=pd.DataFrame(out);table.to_csv(a.out/'delong_signed_auc_contrasts.csv',index=False)
if __name__=='__main__':main()
