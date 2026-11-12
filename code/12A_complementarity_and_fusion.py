#!/usr/bin/env python3
"""Exploratory probability-level fusion with prespecified strategies.

Input base probabilities are fixed seed-averaged OOF scores from the current
analysis. Selection of the FM arm and meta-model fitting are done
only inside each second-stage training fold.
"""
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

MODELS=["Merlin","RADAR","TotalFM","Jolia","OrganLens","Radiomics"]
FMS=MODELS[:-1]
TASKS=[("Steatosis","steatosis"),("NAS >= 3","nas3"),("Sinusoidal dilatation","sinusoidal_dilatation")]
SEEDS=[42,0,1,7,2024]; C_GRID=[.01,.1,1,10]

def midrank(x):
    ix=np.argsort(x); z=x[ix]; r=np.zeros(len(x)); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and z[j]==z[i]: j+=1
        r[i:j]=(i+j-1)/2+1; i=j
    out=np.empty(len(x)); out[ix]=r; return out

def delong_p(y,a,b):
    ix=np.argsort(-y); yy=y[ix]; m=int(yy.sum()); n=len(yy)-m; vectors=[]
    for s in [a[ix],b[ix]]:
        tx,ty,tz=midrank(s[:m]),midrank(s[m:]),midrank(s)
        vectors.append(((tz[:m]-tx)/n,1-(tz[m:]-ty)/m))
    cov=np.atleast_2d(np.cov(np.vstack([v[0] for v in vectors])))/m
    cov+=np.atleast_2d(np.cov(np.vstack([v[1] for v in vectors])))/n
    v=float(cov[0,0]+cov[1,1]-2*cov[0,1]); d=roc_auc_score(y,a)-roc_auc_score(y,b)
    return float(2*norm.sf(abs(d/np.sqrt(v)))) if v>0 else np.nan

def boot(y, score, n, seed):
    rng=np.random.default_rng(seed); pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0); z=np.empty(n)
    for i in range(n):
        ix=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]
        z[i]=roc_auc_score(y[ix],score[ix])
    return np.quantile(z,[.025,.975])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--oof-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--bootstrap',type=int,default=1000); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    summary=[]; all_preds={}; selection=[]; pairrows=[]
    with threadpool_limits(limits=1):
      for ti,(task,slug) in enumerate(TASKS):
        path=a.oof_dir/f'oof_{slug}.csv'
        if not path.exists(): path=a.oof_dir/f'main_oof_{slug}.csv'
        d=pd.read_csv(path,dtype={'case_id':str}); y=d.y_true.to_numpy(int); base=d[MODELS].to_numpy(float); rad_idx=MODELS.index('Radiomics')
        # Cross-fit the two radiomics-anchored ensembles and Meta6.
        byseed={k:[] for k in ['Rad_FM_best','Rad_FM_lowestRho','Meta6']}
        for seed in SEEDS:
          outpred={k:np.full(len(y),np.nan) for k in byseed}; outer=StratifiedKFold(5,shuffle=True,random_state=seed)
          for fold,(tr,te) in enumerate(outer.split(base,y),1):
            aucs=[roc_auc_score(y[tr],base[tr,j]) for j in range(len(FMS))]
            best_ix=max(range(len(FMS)),key=lambda j:(aucs[j],-j)); best=FMS[best_ix]
            rhos=[spearmanr(base[tr,rad_idx],base[tr,j]).statistic for j in range(len(FMS))]
            low_ix=min(range(len(FMS)),key=lambda j:(rhos[j],j)); low=FMS[low_ix]
            outpred['Rad_FM_best'][te]=.5*(base[te,rad_idx]+base[te,best_ix])
            outpred['Rad_FM_lowestRho'][te]=.5*(base[te,rad_idx]+base[te,low_ix])
            selection.append({'Task':task,'Seed':seed,'Fold':fold,'Best_FM':best,'Best_FM_train_AUC':aucs[best_ix],
                              'Lowest_rho_FM':low,'Lowest_rho_with_Radiomics':rhos[low_ix]})
            inner=StratifiedKFold(3,shuffle=True,random_state=seed*100+fold)
            model=make_pipeline(StandardScaler(),LogisticRegression(penalty='l2',solver='lbfgs',max_iter=5000,tol=1e-4))
            grid=GridSearchCV(model,{'logisticregression__C':C_GRID},scoring='roc_auc',cv=inner,n_jobs=1,refit=True)
            grid.fit(base[tr],y[tr]); outpred['Meta6'][te]=grid.predict_proba(base[te])[:,1]
          for name in byseed: byseed[name].append(outpred[name])
        # The final Figure 5C/Table S11C report only radiomics-plus-one-FM
        # Equal-weight pairwise radiomics-plus-FM fusions are reported below.
        strategies={k:np.vstack(v).mean(axis=0) for k,v in byseed.items()}
        all_preds[slug]=d[['case_id','y_true']].copy()
        for k,v in strategies.items(): all_preds[slug][k]=v
        best_single=MODELS[int(np.argmax([roc_auc_score(y,base[:,i]) for i in range(len(MODELS))]))]
        best_score=base[:,MODELS.index(best_single)]
        for si,(name,score) in enumerate(strategies.items()):
            lo,hi=boot(y,score,a.bootstrap,52000+ti*100+si)
            summary.append({'Task':task,'Strategy':name,'N':len(y),'Events':int(y.sum()),'AUC':roc_auc_score(y,score),
                            'AUC_CI_low':lo,'AUC_CI_high':hi,'Brier_score':brier_score_loss(y,score),
                            'Best_single_arm':best_single,'Best_single_AUC':roc_auc_score(y,best_score),
                            'Delta_AUC_vs_best_single':roc_auc_score(y,score)-roc_auc_score(y,best_score),
                            'DeLong_p_vs_best_single':delong_p(y,score,best_score)})
        # Pairwise OOF complementarity retained for the six arms.
        for i,j in combinations(range(len(MODELS)),2):
            pairrows.append({'Task':task,'Model_1':MODELS[i],'Model_2':MODELS[j],
                             'Spearman_rho_OOF_probability':spearmanr(base[:,i],base[:,j]).statistic,
                             'Equal_pair_AUC':roc_auc_score(y,.5*(base[:,i]+base[:,j]))})
    pd.DataFrame(summary).to_csv(a.out/'fusion_performance_raw.csv',index=False)
    pd.DataFrame(pairrows).to_csv(a.out/'pairwise_complementarity_raw.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__': main()
