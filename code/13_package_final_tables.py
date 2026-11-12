#!/usr/bin/env python3
"""Build the manuscript CSV tables from a fresh replay's analysis outputs."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

MODELS=['Merlin','RADAR','TotalFM','Jolia','OrganLens','Radiomics']
TASKS=['Steatosis','NAS >= 3','Sinusoidal dilatation']
def task(x): return 'NAS ≥ 3' if x=='NAS >= 3' else x
def f(x): return '' if pd.isna(x) else f'{float(x):.3f}'
def p(x): return '' if pd.isna(x) else ('<0.001' if float(x)<.001 else f'{float(x):.3f}')
def ci(v,l,h): return f'{float(v):.3f} ({float(l):.3f}-{float(h):.3f})'
def delta_bootstrap(y, fusion, radiomics, n=1000, seed=20261109):
 rng=np.random.default_rng(seed); pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0); vals=[]
 for _ in range(n):
  ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
  vals.append(roc_auc_score(y[ix],fusion[ix])-roc_auc_score(y[ix],radiomics[ix]))
 return np.quantile(vals,[.025,.975])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();A=a.analysis;O=a.out;O.mkdir(parents=True,exist_ok=True)
 seed=pd.read_csv(A/'02_delong/primary_mean_seed_level_auc.csv'); perf=pd.read_csv(A/'01_unified_linear_probe/performance_raw.csv');d=seed.merge(perf[['Task','Model','OOF_AUC','AUC_CI_low','AUC_CI_high']],on=['Task','Model']);d['Task']=d.Task.map(task);d['Mean seed AUC ± SD']=d.apply(lambda r:f'{r.Mean_seed_AUC:.3f} ± {r.SD_seed_AUC:.3f}',axis=1);d['OOF AUC (95% CI; 1000 bootstrap)']=d.apply(lambda r:ci(r.OOF_AUC,r.AUC_CI_low,r.AUC_CI_high),axis=1);d=d.set_index(['Task','Model']).loc[pd.MultiIndex.from_product([[task(x) for x in TASKS],MODELS])].reset_index();d[['Task','Model','N','Events','Mean seed AUC ± SD','OOF AUC (95% CI; 1000 bootstrap)']].to_csv(O/'Table_3_discriminative_performance.csv',index=False)
 cal=pd.read_csv(A/'05_calibration/current_main_nested_calibration_summary.csv').merge(perf[['Task','Model','Brier_score','Brier_CI_low','Brier_CI_high']],on=['Task','Model']);cal['Task']=cal.Task.map(task);cal['Null-model Brier score']=((cal.Events/cal.N)*(1-cal.Events/cal.N)).map(f);cal['Raw Brier score']=cal.Brier_raw_seed_averaged_OOF.map(f);cal['Scaled Brier']=cal.Scaled_Brier_seed_averaged_OOF.map(f);cal['Nested Platt Brier score']=cal.Brier_nested_Platt_seed_averaged_OOF.map(f);cal['Nested Platt calibration intercept']=cal.Mean_nested_Platt_intercept.map(f);cal['Nested Platt calibration slope']=cal.Mean_nested_Platt_slope.map(f);cal=cal.set_index(['Task','Model']).loc[pd.MultiIndex.from_product([[task(x) for x in TASKS],MODELS])].reset_index();cal[['Task','Model','Null-model Brier score','Raw Brier score','Scaled Brier','Nested Platt Brier score','Nested Platt calibration intercept','Nested Platt calibration slope']].to_csv(O/'Table_4_calibration_metrics.csv',index=False)
 fm=pd.read_csv(A/'02_delong/delong_signed_auc_contrasts.csv').query("Family == 'FM_vs_Radiomics'").copy();fm['Comparison family']='FM vs Radiomics';fm['Holm-adjusted p']=fm.Holm_p_within_5_FM_vs_Radiomics;fm=fm.rename(columns={'Signed_delta_AUC':'Signed ΔAUC','DeLong_p':'DeLong p'})[['Task','Comparison family','Baseline','Comparator','Signed ΔAUC','DeLong p','Holm-adjusted p']]
 bl=pd.read_csv(A/'06_simple_baselines/simple_baseline_vs_models_delong.csv').copy();bl['Comparison family']='Model/Radiomics vs imaging baseline';bl['Signed ΔAUC']=bl.Signed_delta_AUC_model_minus_baseline;bl['DeLong p']=bl.DeLong_p;bl['Holm-adjusted p']=bl.Holm_p_within_baseline_task_six_models;bl=bl.rename(columns={'Baseline':'Baseline'})[['Task','Comparison family','Baseline','Comparator','Signed ΔAUC','DeLong p','Holm-adjusted p']];d=pd.concat([fm,bl]);d['Task']=d.Task.map(task);d['Signed ΔAUC']=d['Signed ΔAUC'].map(f);d['DeLong p']=d['DeLong p'].map(p);d['Holm-adjusted p']=d['Holm-adjusted p'].map(p);d.to_csv(O/'Table_S3_pairwise_DeLong_comparisons.csv',index=False)
 d=pd.read_csv(A/'04_permutation_fixed_summary/permutation_18cells_1000_raw.csv');d['Task']=d.Task.map(task);out=pd.DataFrame({'Task':d.Task,'Model':d.Model,'N':d.N,'Events':d.Events,'Observed OOF AUC':d.Observed_OOF_AUC.map(f),'Permutations':d.N_permutations,'Null AUC mean ± SD':d.apply(lambda r:f'{r.Null_AUC_mean:.3f} ± {r.Null_AUC_SD:.3f}',axis=1),'Empirical p':d.Empirical_upper_tail_p.map(p),'Monte Carlo exceedance 95% CI':d.apply(lambda r:f'{r.Monte_Carlo_exceedance_CI_low:.3f}-{r.Monte_Carlo_exceedance_CI_high:.3f}',axis=1),'Within-task Holm p':d.Holm_p_within_task.map(p)});out.to_csv(O/'Table_S4_permutation_tests_18_cells.csv',index=False)
 d=pd.read_csv(A/'03_threshold_metrics/threshold_metrics_patient_bootstrap_raw.csv');w=d.pivot(index=['Task','Model','N','Events'],columns='Metric',values=['Estimate','CI_low','CI_high']).reset_index();w.columns=['_'.join(str(x) for x in c if x) for c in w.columns];out=w[['Task','Model','N','Events']].copy();out['Task']=out.Task.map(task)
 for m in ['Sensitivity','Specificity','PPV','NPV']:out[f'{m} (95% CI)']=w.apply(lambda r:ci(r[f'Estimate_{m}'],r[f'CI_low_{m}'],r[f'CI_high_{m}']),axis=1)
 out.to_csv(O/'Table_S5_threshold_derived_metrics.csv',index=False)
 for tag,pre,pp in [('A','08_totalfm_roi','totalfm_roi'),('B','09_radiomics_roi','radiomics_roi')]:
  d=pd.read_csv(A/pre/f'{pp}_performance_raw.csv');d['Task']=d.Task.map(task);d['OOF AUC (95% CI)']=d.apply(lambda r:ci(r.get('AUC_seed_averaged_OOF',r.get('OOF_AUC')),r.AUC_CI_low,r.AUC_CI_high),axis=1)
  if tag=='A':
   d['Mean seed AUC ± SD']=d.apply(lambda r:f'{r.Mean_seed_AUC:.3f} ± {r.SD_seed_AUC:.3f}',axis=1);d[['Task','ROI','N','Events','Mean seed AUC ± SD','OOF AUC (95% CI)']].to_csv(O/'Table_S6A_TotalFM_VOI_performance.csv',index=False)
   q=pd.read_csv(A/pre/f'{pp}_pairwise_delong_raw.csv');q['Task']=q.Task.map(task);q=q.rename(columns={'Baseline_ROI':'Baseline VOI','Comparator_ROI':'Comparator VOI','Signed_delta_AUC_comparator_minus_baseline':'Signed ΔAUC','DeLong_p':'DeLong p','Holm_p_within_task_three_ROI_pairs':'Holm p'});q['Signed ΔAUC']=q['Signed ΔAUC'].map(f);q['DeLong p']=q['DeLong p'].map(p);q['Holm p']=q['Holm p'].map(p);q[['Task','Baseline VOI','Comparator VOI','Signed ΔAUC','DeLong p','Holm p']].to_csv(O/'Table_S6B_TotalFM_VOI_DeLong.csv',index=False)
  else:
   d['Mean seed AUC ± SD']=d.apply(lambda r:f'{r.Mean_seed_AUC:.3f} ± {r.SD_seed_AUC:.3f}',axis=1)
   d=d.rename(columns={'Radiomics_set':'VOI'});d[['Task','VOI','N','Positive','Features','Mean seed AUC ± SD','OOF AUC (95% CI)']].to_csv(O/'Table_S7A_Radiomics_VOI_performance.csv',index=False)
   q=pd.read_csv(A/pre/f'{pp}_pairwise_delong_raw.csv');q['Task']=q.Task.map(task);q=q.rename(columns={'ROI_1':'VOI 1','ROI_2':'VOI 2','Signed_delta_AUC_ROI2_minus_ROI1':'Signed ΔAUC','DeLong_p':'DeLong p','Holm_p_within_task':'Holm p'});q['Signed ΔAUC']=q['Signed ΔAUC'].map(f);q['DeLong p']=q['DeLong p'].map(p);q['Holm p']=q['Holm p'].map(p);q[['Task','VOI 1','VOI 2','Signed ΔAUC','DeLong p','Holm p']].to_csv(O/'Table_S7B_Radiomics_VOI_DeLong.csv',index=False)
 c=pd.read_excel(a.data/'Colorectal-Liver-Metastases-Clinical-data-April-2023.xlsx',sheet_name='TCIA_n=197');s=pd.to_numeric(c.steatosis_yesno);n=(pd.to_numeric(c.NASH_score)>=3).astype(int);ch=pd.to_numeric(c.chemo_before_liver_resection).astype(int)
 rows=[]
 for label,v in [('NAS < 3',0),('NAS ≥ 3',1)]:x=s[n.eq(v)];rows.append({'NAS category':label,'Steatosis absent':int((x==0).sum()),'Steatosis present':int((x==1).sum()),'n':len(x),'Steatosis present (%)':f'{100*(x==1).mean():.1f}'})
 pd.DataFrame(rows).to_csv(O/'Table_S8A_steatosis_by_NAS3_crosstab.csv',index=False)
 d=pd.read_csv(A/'10_nas3_positive/nas3_steatosis_positive_auc.csv');d['Mean seed AUC ± SD']=d.apply(lambda r:f'{r.Mean_seed_AUC:.3f} ± {r.SD_seed_AUC:.3f}',axis=1);d['OOF AUC (95% CI)']=d.apply(lambda r:ci(r.AUC_seed_averaged_OOF,r.AUC_CI_low,r.AUC_CI_high),axis=1);d[['Model','N','Events','Mean seed AUC ± SD','OOF AUC (95% CI)']].to_csv(O/'Table_S8B_NAS3_steatosis_positive.csv',index=False)
 rows=[]
 for name,y in [('Steatosis',s.astype(int)),('NAS ≥ 3',n),('Sinusoidal dilatation',pd.to_numeric(c.presence_sinusoidal_dilata).astype(int))]:
  for label,v in [('Negative',0),('Positive',1)]:a0=int(((ch==0)&(y==v)).sum());a1=int(((ch==1)&(y==v)).sum());rows.append({'Task':name,'Outcome':label,'No chemotherapy n':a0,'No chemotherapy %':f'{100*a0/(ch==0).sum():.1f}','Chemotherapy n':a1,'Chemotherapy %':f'{100*a1/(ch==1).sum():.1f}','Overall n':a0+a1,'Overall %':f'{100*(a0+a1)/len(y):.1f}'})
 pd.DataFrame(rows).to_csv(O/'Table_S9A_chemotherapy_by_task_crosstab.csv',index=False)
 d=pd.read_csv(A/'11_chemotherapy/subgroup_auc_raw.csv');d['Task']=d.Task.map(task);d['AUC (95% CI)']=d.apply(lambda r:ci(r.AUC,r.CI_low,r.CI_high),axis=1);d[['Task','Model','Chemo_status','N','Events','AUC (95% CI)']].to_csv(O/'Table_S9B_chemotherapy_subgroup.csv',index=False)
 pair=pd.read_csv(A/'12_fusion/pairwise_complementarity_raw.csv')
 pair=pair[((pair.Model_1!='Radiomics') & (pair.Model_2=='Radiomics')) | ((pair.Model_2!='Radiomics') & (pair.Model_1=='Radiomics'))].copy()
 pair['FM']=pair.apply(lambda r: r.Model_1 if r.Model_1!='Radiomics' else r.Model_2,axis=1)
 oof_dir=A/'01_unified_linear_probe'; rows=[]
 for task_name,slug in [('Steatosis','steatosis'),('NAS >= 3','nas3'),('Sinusoidal dilatation','sinusoidal_dilatation')]:
  q=pd.read_csv(oof_dir/f'oof_{slug}.csv'); rad_auc=roc_auc_score(q.y_true,q.Radiomics)
  for _,r in pair[pair.Task.eq(task_name)].iterrows():
   fm_auc=roc_auc_score(q.y_true,q[r.FM])
   fusion=.5*(q[r.FM].to_numpy(float)+q.Radiomics.to_numpy(float))
   dlo,dhi=delta_bootstrap(q.y_true.to_numpy(int),fusion,q.Radiomics.to_numpy(float),seed=20261109+len(rows))
   delta=roc_auc_score(q.y_true,fusion)-rad_auc
   rows.append({'Task':task_name,'FM':r.FM,'FM OOF AUC':f'{fm_auc:.3f}',
                'Radiomics OOF AUC':f'{rad_auc:.3f}',
                'Equal-weight fusion AUC':f'{roc_auc_score(q.y_true,fusion):.3f}',
                'ΔAUC vs Radiomics (95% CI)':f'{delta:.3f} ({dlo:.3f}-{dhi:.3f})'})
 late=pd.DataFrame(rows).assign(Task=lambda d:d.Task.map(task))
 pair=pd.read_csv(A/'12_fusion/pairwise_complementarity_raw.csv');pair=pair[pair.Model_2.eq('Radiomics')].rename(columns={'Model_1':'FM','Model_2':'Reference'});pair['Task']=pair.Task.map(task)
 spearman_rows=[]
 for task_name,slug in [('Steatosis','steatosis'),('NAS >= 3','nas3'),('Sinusoidal dilatation','sinusoidal_dilatation')]:
  q=pd.read_csv(A/'01_unified_linear_probe'/f'oof_{slug}.csv')
  for fm in pair.loc[pair.Task.eq(task(task_name)),'FM']:
   rho,pv=spearmanr(q[fm],q.Radiomics)
   spearman_rows.append({'Task':task_name,'FM':fm,'Reference':'Radiomics','Spearman ρ':f'{rho:.3f}','p value':('<0.001' if pv<.001 else f'{pv:.3f}')})
 corr=pd.DataFrame(spearman_rows);corr['Task']=corr.Task.map(task)
 ov=pd.read_csv(A/'05_calibration/current_main_threshold_error_overlap.csv');ov=ov[ov.Model_2.eq('Radiomics')].rename(columns={'Model_1':'FM','Errors_model_1':'Errors FM','Errors_model_2':'Errors Radiomics','Both_wrong':'Both wrong'});ov['Task']=ov.Task.map(task)
 agreement=corr.merge(ov[['Task','FM','Errors FM','Errors Radiomics','Both wrong','Error_Jaccard']],on=['Task','FM'],how='left')
 agreement['Error_Jaccard']=agreement['Error_Jaccard'].map(f)
 agreement.to_csv(O/'Table_S10A_prediction_agreement_error_overlap.csv',index=False)
 late.to_csv(O/'Table_S10B_late_fusion_performance.csv',index=False)

if __name__=='__main__':main()
