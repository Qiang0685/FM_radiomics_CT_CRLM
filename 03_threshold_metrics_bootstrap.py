#!/usr/bin/env python3
"""Patient-bootstrap CIs for fold-internally selected threshold metrics."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['Merlin','RADAR','TotalFM','Jolia','OrganLens','Radiomics']
TASKS=[('Steatosis','steatosis'),('NAS ≥ 3','nas3'),('Sinusoidal dilatation','sinusoidal_dilatation')]
METRICS=['Sensitivity','Specificity','PPV','NPV']
METRIC_COLORS=['#4C78A8','#59A14F','#F28E2B','#B279A2']
BOOTSTRAP_SEED=9715

def metric_values(y,p):
    tp=np.sum((y==1)&(p==1)); fn=np.sum((y==1)&(p==0)); tn=np.sum((y==0)&(p==0)); fp=np.sum((y==0)&(p==1))
    return np.array([tp/(tp+fn) if tp+fn else np.nan, tn/(tn+fp) if tn+fp else np.nan,
                     tp/(tp+fp) if tp+fp else np.nan, tn/(tn+fn) if tn+fn else np.nan])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--oof-dir',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--figure-dir',type=Path,required=True)
    ap.add_argument('--bootstrap',type=int,default=1000); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); a.figure_dir.mkdir(parents=True,exist_ok=True)
    if a.bootstrap != 1000:
        raise ValueError('This study uses exactly 1,000 bootstrap resamples')
    rows=[]; rng=np.random.default_rng(BOOTSTRAP_SEED)
    for task,slug in TASKS:
        path=a.oof_dir/f'nested_predictions_{slug}.csv'
        if not path.exists(): path=a.oof_dir/'nested_current_main'/f'nested_predictions_{slug}.csv'
        d=pd.read_csv(path,dtype={'case_id':str})
        y=d.y_true.to_numpy(int)
        pos,neg=np.flatnonzero(y==1),np.flatnonzero(y==0)
        for model in MODELS:
            p=d[f'{model}_majority_prediction'].to_numpy(int)
            point=metric_values(y,p); draws=np.empty((a.bootstrap,4))
            for b in range(a.bootstrap):
                ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
                draws[b]=metric_values(y[ix],p[ix])
            low,high=np.nanquantile(draws,[.025,.975],axis=0)
            for j,m in enumerate(METRICS):
                rows.append({'Task':task,'Model':model,'N':len(y),'Events':int(y.sum()),'Metric':m,
                             'Estimate':point[j],'CI_low':low[j],'CI_high':high[j],
                             'Prediction_rule':'Majority vote across five seed-specific outer-fold predictions; each threshold selected by Youden index within outer training data'})
    result=pd.DataFrame(rows); result.to_csv(a.out/'threshold_metrics_patient_bootstrap_raw.csv',index=False)
    colors={'Merlin':'#4C78A8','RADAR':'#59A14F','TotalFM':'#F28E2B','Jolia':'#B279A2','OrganLens':'#76B7B2','Radiomics':'#7F7F7F'}
    fig,axes=plt.subplots(3,1,figsize=(180/25.4,145/25.4),sharex=True)
    x=np.arange(len(MODELS)); w=.18; offsets=(np.arange(4)-1.5)*w
    for ax,(task,_) in zip(axes,TASKS):
        sub=result[result.Task==task]
        for j,m in enumerate(METRICS):
            s=sub[sub.Metric==m].set_index('Model').loc[MODELS]
            vals=s.Estimate.to_numpy(float); lo=s.CI_low.to_numpy(float); hi=s.CI_high.to_numpy(float)
            ax.bar(x+offsets[j],vals,width=w,color=METRIC_COLORS[j],edgecolor='white',linewidth=.3,label=m,zorder=3,
                   yerr=np.vstack((vals-lo,hi-vals)),error_kw={'ecolor':'#30343B','elinewidth':.7,'capsize':1.7,'capthick':.7})
        ax.set_ylim(0,1.04); ax.set_ylabel('Metric value'); ax.set_title(task,loc='left',fontsize=8)
        ax.grid(axis='y',color='#E1E4E8',linewidth=.5); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xticks(x,MODELS,rotation=25,ha='right')
    h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,ncol=4,loc='upper center',bbox_to_anchor=(.52,.995),frameon=False)
    fig.subplots_adjust(left=.08,right=.99,bottom=.12,top=.93,hspace=.37)
    fig.savefig(a.figure_dir/'Figure_S2_threshold_derived_metrics.pdf',facecolor='white')
    fig.savefig(a.figure_dir/'Figure_S2_threshold_derived_metrics.png',dpi=400,facecolor='white'); plt.close(fig)
    print(f'Saved patient-bootstrap threshold metric CIs (bootstrap seed={BOOTSTRAP_SEED})')

if __name__=='__main__': main()
