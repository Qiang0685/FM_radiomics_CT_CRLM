#!/usr/bin/env python3
"""Run remaining feasible nested calibration, threshold, and seed-stability checks.

Uses the exact five outer seeds, inner folds, C grid, scaling, classifier, and
six-arm feature matrices of the current unified linear-probe analysis. This is
an additional CV run for previously unreported analyses; it does not replace
the already archived main OOF outputs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEEDS=(42,0,1,7,2024)
C_GRID=(.01,.1,1.,10.)
TASKS=(("Steatosis","steatosis"),("NAS >= 3","nas3"),("Sinusoidal dilatation","sinusoidal_dilatation"))
MODELS=("Merlin","RADAR","TotalFM","Jolia","OrganLens","Radiomics")
EPS=1e-6

def read_features(path, ids, radar=False):
    d=pd.read_csv(path,dtype={"case_id":str})
    case_col="case_id" if "case_id" in d.columns else d.columns[0]
    ids_ex=d[case_col].astype(str).str.extract(r"(\d{4})",expand=False)
    if ids_ex.isna().any() or ids_ex.duplicated().any(): raise ValueError(f"Invalid IDs in {path}")
    d.index=ids_ex
    if radar:
        cols=[c for c in d.columns if c.startswith("liver_")]
    else: cols=[c for c in d.columns if c != case_col]
    d=d.loc[list(ids),cols]
    x=d.to_numpy(float)
    if not np.isfinite(x).all(): raise ValueError(f"Nonfinite values in {path}")
    return x

def make_model(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c,penalty="l2",solver="lbfgs",max_iter=5000,tol=1e-4))

def youden_threshold(y,p):
    fpr,tpr,thr=roc_curve(y,p)
    j=tpr-fpr
    valid=np.isfinite(thr)
    idx=np.flatnonzero(valid)[np.argmax(j[valid])]
    return float(thr[idx])

def fit_platt(y,p):
    z=logit(np.clip(p,EPS,1-EPS)).reshape(-1,1)
    cal=LogisticRegression(C=1e6,solver="lbfgs",max_iter=5000)
    cal.fit(z,y)
    return float(cal.intercept_[0]),float(cal.coef_[0,0])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',type=Path,required=True,
                    help='Direct Data directory containing the six feature matrices.')
    ap.add_argument('--oof-dir',type=Path,required=True,
                    help='Directory containing main_oof_{steatosis,nas3,sinusoidal_dilatation}.csv')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); data_root=args.data_root; out=args.out
    out.mkdir(parents=True,exist_ok=True)
    main_oof={}
    for _,slug in TASKS:
        main_path=args.oof_dir/f'main_oof_{slug}.csv'
        if not main_path.exists(): main_path=args.oof_dir/f'oof_{slug}.csv'
        main_oof[slug]=pd.read_csv(main_path)
    fmap={
      'Merlin':(data_root/'01_Merlin_197x2048.csv',False),
      'RADAR':(data_root/'02_RADAR_liver_embeddings_197x256.csv',True),
      'TotalFM':(data_root/'03_TotalFM_liver_TS_197x768.csv',False),
      'Jolia':(data_root/'04_Jolia_liver_197x576.csv',False),
      'OrganLens':(data_root/'05_OrganLens_liver_197x1024.csv',False),
      'Radiomics':(data_root/'06_radiomics_resected_TumorFree_197x1130.csv',False),
    }
    all_seed_rows=[]; calibration_rows=[]; threshold_rows=[]; overlap_rows=[]
    with threadpool_limits(limits=1):
      for task,slug in TASKS:
        d=main_oof[slug]; ids=d.case_id.astype(str).tolist(); y=d.y_true.to_numpy(int)
        taskpred=pd.DataFrame({'case_id':ids,'y_true':y})
        for model,(path,isradar) in fmap.items():
          X=read_features(path,ids,isradar)
          seed_raw=[]; seed_cal=[]; seed_hard=[]
          cms=[]
          for seed in SEEDS:
            # Keep fold-level calibration summaries independent for each seed.
            # Otherwise later seed rows would contain cumulative means.
            fold_thresholds=[]; cal_slopes=[]; cal_intercepts=[]
            p_raw=np.full(len(y),np.nan); p_cal=np.full(len(y),np.nan)
            hard=np.full(len(y),-1,dtype=int)
            test_y=[]; test_pred=[]
            outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
            for fold,(tr,te) in enumerate(outer.split(X,y),1):
              inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed*100+fold)
              inner_splits=list(inner.split(X[tr],y[tr]))
              scores=[]; fold_probs=[]
              for c in C_GRID:
                oof_inner=np.full(len(tr),np.nan); fold_auc=[]
                for itr,iva in inner_splits:
                  mdl=make_model(c)
                  mdl.fit(X[tr][itr],y[tr][itr])
                  pp=mdl.predict_proba(X[tr][iva])[:,1]
                  oof_inner[iva]=pp
                  fold_auc.append(roc_auc_score(y[tr][iva],pp))
                scores.append(float(np.mean(fold_auc))); fold_probs.append(oof_inner)
              best=int(np.argmax(scores)); c=C_GRID[best]; inner_p=fold_probs[best]
              if not np.isfinite(inner_p).all(): raise RuntimeError('inner OOF incomplete')
              thr=youden_threshold(y[tr],inner_p)
              intercept,slope=fit_platt(y[tr],inner_p)
              cal_slopes.append(slope); cal_intercepts.append(intercept); fold_thresholds.append(thr)
              model_final=make_model(c); model_final.fit(X[tr],y[tr])
              pp=model_final.predict_proba(X[te])[:,1]
              pc=expit(intercept+slope*logit(np.clip(pp,EPS,1-EPS)))
              p_raw[te]=pp; p_cal[te]=pc
              hard[te]=(pp>=thr).astype(int)
              test_y.extend(y[te].tolist()); test_pred.extend((pp>=thr).astype(int).tolist())
            if not np.isfinite(p_raw).all() or not np.isfinite(p_cal).all() or (hard<0).any():
                raise RuntimeError('outer OOF incomplete')
            seed_raw.append(p_raw); seed_cal.append(p_cal); seed_hard.append(hard)
            all_seed_rows.append({'Task':task,'Model':model,'Seed':seed,'N':len(y),'Events':int(y.sum()),
                                  'AUC_raw_seed_OOF':roc_auc_score(y,p_raw),'Brier_raw_seed_OOF':brier_score_loss(y,p_raw),
                                  'Brier_nested_Platt_seed_OOF':brier_score_loss(y,p_cal),
                                  'Mean_train_Youden_threshold':float(np.mean(fold_thresholds)),
                                  'Mean_inner_Platt_intercept':float(np.mean(cal_intercepts)),
                                  'Mean_inner_Platt_slope':float(np.mean(cal_slopes))})
            cms.append(confusion_matrix(np.asarray(test_y),np.asarray(test_pred),labels=[0,1]))
          p_raw_mean=np.mean(seed_raw,axis=0); p_cal_mean=np.mean(seed_cal,axis=0)
          taskpred[model]=p_raw_mean
          taskpred[f'{model}_nested_Platt']=p_cal_mean
          taskpred[f'{model}_majority_prediction']=(np.sum(seed_hard,axis=0)>=3).astype(int)
          cmsum=np.sum(cms,axis=0); tn,fp,fn,tp=cmsum.ravel()
          null_brier=float(y.mean()*(1-y.mean()))
          sens=tp/(tp+fn) if tp+fn else np.nan
          spec=tn/(tn+fp) if tn+fp else np.nan
          ppv=tp/(tp+fp) if tp+fp else np.nan
          npv=tn/(tn+fn) if tn+fn else np.nan
          seed_df=pd.DataFrame([r for r in all_seed_rows if r['Task']==task and r['Model']==model])
          calibration_rows.append({'Task':task,'Model':model,'N':len(y),'Events':int(y.sum()),
             'Brier_raw_seed_averaged_OOF':brier_score_loss(y,p_raw_mean),
             'Scaled_Brier_seed_averaged_OOF':1-brier_score_loss(y,p_raw_mean)/null_brier,
             'Brier_nested_Platt_seed_averaged_OOF':brier_score_loss(y,p_cal_mean),
             'AUC_raw_seed_averaged_OOF':roc_auc_score(y,p_raw_mean),
             'AUC_nested_Platt_seed_averaged_OOF':roc_auc_score(y,p_cal_mean),
             'Mean_nested_Platt_intercept':float(seed_df[(seed_df.Task==task)&(seed_df.Model==model)]['Mean_inner_Platt_intercept'].mean()),
             'Mean_nested_Platt_slope':float(seed_df[(seed_df.Task==task)&(seed_df.Model==model)]['Mean_inner_Platt_slope'].mean()),
             'Note':'Each outer-fold Platt model trained on inner OOF predictions; held-out predictions aggregated over five outer seeds.'})
          threshold_rows.append({'Task':task,'Model':model,'N_repeated_test_predictions':5*len(y),'Events_repeated_test_predictions':5*int(y.sum()),
             'Sensitivity':sens,'Specificity':spec,'PPV':ppv,'NPV':npv,'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
             'Threshold_selection':'Youden index on best-C inner OOF predictions within each outer training set; applied to outer held-out fold.'})
        for m1,m2 in itertools.combinations(MODELS,2):
            p1=taskpred[f'{m1}_majority_prediction'].to_numpy(int)
            p2=taskpred[f'{m2}_majority_prediction'].to_numpy(int)
            e1,e2=p1!=y,p2!=y
            either=int(np.sum(e1|e2)); both=int(np.sum(e1&e2))
            overlap_rows.append({'Task':task,'Model_1':m1,'Model_2':m2,'N':len(y),
                'Errors_model_1':int(e1.sum()),'Errors_model_2':int(e2.sum()),
                'Both_wrong':both,'Either_wrong':either,
                'Error_Jaccard':both/either if either else np.nan,
                'Prediction_disagreement_fraction':float(np.mean(p1!=p2))})
        taskpred.to_csv(out/f'nested_predictions_{slug}.csv',index=False)
    seed_df=pd.DataFrame(all_seed_rows)
    seed_df.to_csv(out/'seed_level_performance_nested_calibration.csv',index=False)
    pd.DataFrame(calibration_rows).to_csv(out/'current_main_nested_calibration_summary.csv',index=False)
    pd.DataFrame(overlap_rows).to_csv(out/'current_main_threshold_error_overlap.csv',index=False)
    (out/'nested_run_manifest.json').write_text(json.dumps({'seeds':list(SEEDS),'outer':'5-fold stratified, shuffle=True',
       'inner':'3-fold stratified with random_state=seed*100+outer_fold','C_grid':list(C_GRID),
       'models':list(MODELS),'tasks':[x[0] for x in TASKS],'scaling':'StandardScaler within each training fold',
       'classifier':'L2 LogisticRegression, lbfgs','threshold':'Youden threshold on inner OOF probabilities at selected C, evaluated in outer held-out fold',
       'calibration':'Platt logistic regression fitted to inner OOF logit probabilities at selected C, evaluated in outer held-out fold',
       'note':'Additional complete nested-CV fits. Output supplements and does not overwrite archived primary OOF results.'},indent=2),encoding='utf-8')
    print('completed nested stability/calibration/threshold analysis:',out)

if __name__=='__main__': main()
