#!/usr/bin/env python3
"""Nested-CV NAS>=3 analysis restricted to steatosis-positive cases."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

MODELS = ["Merlin", "RADAR", "TotalFM", "Jolia", "OrganLens", "Radiomics"]
FILES = {
    "Merlin": "01_Merlin_197x2048.csv",
    "RADAR": "02_RADAR_liver_embeddings_197x256.csv",
    "TotalFM": "03_TotalFM_liver_TS_197x768.csv",
    "Jolia": "04_Jolia_liver_197x576.csv",
    "OrganLens": "05_OrganLens_liver_197x1024.csv",
    "Radiomics": "06_radiomics_resected_TumorFree_197x1130.csv",
}
COLORS = {"Merlin":"#4C78A8", "RADAR":"#59A14F", "TotalFM":"#F28E2B",
          "Jolia":"#B279A2", "OrganLens":"#76B7B2", "Radiomics":"#7F7F7F"}
SEEDS = [42, 0, 1, 7, 2024]
C_GRID = [0.01, 0.1, 1, 10]

def read_matrix(path, ids, is_radar=False):
    d = pd.read_csv(path, dtype={"case_id":str})
    cid = d.case_id.astype(str).str.extract(r"(\d{4})", expand=False)
    if cid.isna().any() or cid.duplicated().any(): raise ValueError(f"Bad IDs in {path}")
    d.index = cid
    cols = [c for c in d if c != "case_id"]
    if is_radar: cols = [c for c in cols if c.startswith("liver_")]
    x = d.loc[ids, cols].to_numpy(float)
    if not np.isfinite(x).all(): raise ValueError(f"Nonfinite features: {path}")
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--analysis-dir", type=Path, required=True,
                    help="Directory for numerical outputs.")
    ap.add_argument("--figure-dir", type=Path, required=True,
                    help="Figure-only output directory (PDF and PNG only).")
    ap.add_argument("--bootstrap", type=int, default=1000)
    a=ap.parse_args(); a.analysis_dir.mkdir(parents=True, exist_ok=True); a.figure_dir.mkdir(parents=True, exist_ok=True)
    stea_path=a.oof/"oof_steatosis.csv"
    nas_path=a.oof/"oof_nas3.csv"
    if not stea_path.exists(): stea_path=a.oof/"main_oof_steatosis.csv"
    if not nas_path.exists(): nas_path=a.oof/"main_oof_nas3.csv"
    stea=pd.read_csv(stea_path,dtype={"case_id":str})
    nas=pd.read_csv(nas_path,dtype={"case_id":str})
    nas=nas.set_index("case_id").loc[stea.case_id].reset_index()
    keep=stea.y_true.to_numpy(int)==1
    ids=stea.case_id[keep].astype(str).str.extract(r"(\d{4})",expand=False).tolist()
    y=nas.y_true.to_numpy(int)[keep]
    if len(y)!=68 or int(y.sum())!=15: raise ValueError(f"Expected 68 steatosis-positive cases/15 events, got {len(y)}/{y.sum()}")
    rows=[]; pred=pd.DataFrame({"case_id":stea.case_id[keep].values,"y_true":y})
    rng=np.random.default_rng(23011)
    with threadpool_limits(limits=1):
        for model in MODELS:
            path=a.data/FILES[model]
            X=read_matrix(path,ids,model=="RADAR")
            seed_scores=[]; fold_c=[]
            for seed in SEEDS:
                p=np.full(len(y),np.nan)
                outer=StratifiedKFold(5,shuffle=True,random_state=seed)
                for fold,(tr,te) in enumerate(outer.split(X,y),1):
                    inner=StratifiedKFold(3,shuffle=True,random_state=seed*100+fold)
                    pipe=make_pipeline(StandardScaler(),LogisticRegression(penalty="l2",solver="lbfgs",max_iter=5000,tol=1e-4))
                    grid=GridSearchCV(pipe,{"logisticregression__C":C_GRID},scoring="roc_auc",cv=inner,n_jobs=1,refit=True)
                    grid.fit(X[tr],y[tr]); p[te]=grid.predict_proba(X[te])[:,1]
                    fold_c.append({"Model":model,"Seed":seed,"Fold":fold,"C":grid.best_params_["logisticregression__C"]})
                if not np.isfinite(p).all(): raise RuntimeError("OOF incomplete")
                seed_scores.append(p)
            ps=np.vstack(seed_scores); pmean=ps.mean(axis=0)
            seed_aucs=[roc_auc_score(y,p) for p in ps]
            auc=roc_auc_score(y,pmean)
            pos,neg=np.flatnonzero(y==1),np.flatnonzero(y==0)
            boots=np.empty(a.bootstrap)
            for b in range(a.bootstrap):
                ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
                boots[b]=roc_auc_score(y[ix],pmean[ix])
            lo,hi=np.quantile(boots,[.025,.975])
            rows.append({"Task":"NAS >= 3 among steatosis-positive cases","Model":model,"N":len(y),"Events":int(y.sum()),
                         "Mean_seed_AUC":float(np.mean(seed_aucs)),"SD_seed_AUC":float(np.std(seed_aucs,ddof=1)),
                         "AUC_seed_averaged_OOF":auc,"AUC_CI_low":float(lo),"AUC_CI_high":float(hi)})
            pred[model]=pmean
    result=pd.DataFrame(rows)
    result.to_csv(a.analysis_dir/"nas3_steatosis_positive_auc.csv",index=False)
    fig,ax=plt.subplots(figsize=(6.8,3.8))
    x=np.arange(len(MODELS)); vals=result.set_index("Model").loc[MODELS]
    ax.bar(x,vals.AUC_seed_averaged_OOF,color=[COLORS[m] for m in MODELS],width=.66,
           yerr=np.vstack([vals.AUC_seed_averaged_OOF-vals.AUC_CI_low,vals.AUC_CI_high-vals.AUC_seed_averaged_OOF]),
           error_kw={"ecolor":"#30343B","elinewidth":1.2,"capsize":3,"capthick":1.2})
    ax.axhline(.5,color="#777777",linestyle="--",linewidth=.9)
    ax.set_xticks(x,MODELS,rotation=25,ha="right"); ax.set_ylabel("OOF AUC")
    ax.set_ylim(0,1.02); ax.grid(axis="y",color="#D9DDE3",linewidth=.55); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(a.figure_dir/"Figure_S5_NAS3_steatosis_positive.pdf"); fig.savefig(a.figure_dir/"Figure_S5_NAS3_steatosis_positive.png",dpi=300)
    plt.close(fig)
    print(f"Completed restricted NAS≥3 analysis: n={len(y)}, events={int(y.sum())}")

if __name__=="__main__": main()
