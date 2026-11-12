from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

task_order=['Steatosis','NAS ≥ 3','Sinusoidal dilatation']
roi_order=['Whole liver with tumor','Whole liver without tumor','Resected tumor-free liver']
rad_roi_order=['Whole liver with tumor','Whole liver without tumor','Resected tumor-free liver']
fm_short=['TotalSegmentator\nliver','TCIA liver','Resected tumor-free\nliver']
rad_short=['Whole liver\nwith tumor','Whole liver\nwithout tumor','Resected tumor-free\nliver']
bar_colors=['#4C78A8','#76B7B2','#7F7F7F']
rad_key_map={'whole_noVes':'Whole liver with tumor',
             'whole_noVesTumor':'Whole liver without tumor',
             'resected_noVesTumor':'Resected tumor-free liver'}
ap=argparse.ArgumentParser()
ap.add_argument('--totalfm-performance',type=Path,required=True); ap.add_argument('--totalfm-delong',type=Path,required=True)
ap.add_argument('--radiomics-performance',type=Path,required=True); ap.add_argument('--radiomics-delong',type=Path,required=True)
ap.add_argument('--out-dir',type=Path,required=True); args=ap.parse_args(); OUT=args.out_dir; OUT.mkdir(parents=True,exist_ok=True)
fm = pd.read_csv(args.totalfm_performance)
rad = pd.read_csv(args.radiomics_performance)
fm['Task']=fm['Task'].replace({'NAS >= 3':'NAS ≥ 3'})
rad['Task']=rad['Task'].replace({'NAS >= 3':'NAS ≥ 3'})
fm['ROI']=fm['ROI'].replace({'TotalSegmentator liver':'Whole liver with tumor','TCIA liver':'Whole liver without tumor','Resected tumor-free liver':'Resected tumor-free liver'})
rad['ROI']=rad['Radiomics_set'].map(rad_key_map)
fm_p=pd.read_csv(args.totalfm_delong); fm_p['Task']=fm_p['Task'].replace({'NAS >= 3':'NAS ≥ 3'})
rad_p=pd.read_csv(args.radiomics_delong); rad_p['Task']=rad_p['Task'].replace({'NAS >= 3':'NAS ≥ 3'})
for col in ('ROI_1','ROI_2'):
    if col in rad_p.columns:
        rad_p[col]=rad_p[col].replace(rad_key_map)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(2,3,figsize=(7.50,5.55),sharey=True)
def p_matrix(df, panel):
    out={}
    fm_name_map={'TotalSegmentator liver':'Whole liver with tumor','TCIA liver':'Whole liver without tumor','Resected tumor-free liver':'Resected tumor-free liver'}
    order = roi_order if panel=='fm' else rad_roi_order
    for task in task_order:
        out[task]={}
        d=df[df.Task==task]
        for _,r in d.iterrows():
            a=r['Baseline_ROI'] if panel=='fm' else r['ROI_1']; b=r['Comparator_ROI'] if panel=='fm' else r['ROI_2']
            if panel=='fm':
                a=fm_name_map.get(a,a); b=fm_name_map.get(b,b)
            p=r['Holm_p_within_task_three_ROI_pairs'] if panel=='fm' else r['Holm_p_within_task']
            valid_pairs=[(x,y) for x,y in [(order[0],order[1]),(order[0],order[2]),(order[1],order[2])]]
            if (a,b) in valid_pairs:
                key=(a,b)
            elif (b,a) in valid_pairs:
                key=(b,a)
            else:
                raise KeyError(f'Unexpected ROI pair in {panel}: {(a,b)}')
            out[task][key]=float(p)
    return out
def draw_row(row_axes, data, pvals, letter, xlabels):
    for j,task in enumerate(task_order):
        ax=row_axes[j]; order = roi_order if letter=='A' else rad_roi_order
        d=data[data.Task==task].set_index('ROI').reindex(order)
        vals=d['AUC_seed_averaged_OOF'].to_numpy(float) if 'AUC_seed_averaged_OOF' in d else d['OOF_AUC'].to_numpy(float)
        lo=d['AUC_CI_low'].to_numpy(float); hi=d['AUC_CI_high'].to_numpy(float)
        x=np.arange(3); err=np.vstack([vals-lo,hi-vals])
        ax.bar(x,vals,color=bar_colors,width=.62,yerr=err,error_kw={'ecolor':'#39414A','elinewidth':1.0,'capsize':3,'capthick':1.0})
        ax.axhline(.5,color='#777',linestyle='--',lw=.8); ax.set_ylim(.35,1.18); ax.set_yticks([.4,.6,.8,1.0]); ax.grid(axis='y',color='#E1E4E8',lw=.5); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
        ax.set_title(task,fontsize=9,pad=14)
        ax.set_xticks(x,xlabels,rotation=35,ha='right',fontsize=6.5); ax.tick_params(axis='y',labelsize=7)
        # Pairwise brackets: 0-1, 0-2, 1-2.
        pairs=[(0,1),(0,2),(1,2)]; y0=1.03
        for k,(a,b) in enumerate(pairs):
            order = roi_order if letter=='A' else rad_roi_order
            key=(order[a],order[b])
            if key not in pvals[task]:
                raise KeyError(f'Missing p value for {task}, pair {key}')
            p=pvals[task][key]; y=y0+k*.065
            ax.plot([a,b],[y,y],color='#59616A',lw=.8); ax.plot([a,a],[y-.012,y],color='#59616A',lw=.8); ax.plot([b,b],[y-.012,y],color='#59616A',lw=.8)
            ax.text((a+b)/2,y+.008,f'p={p:.3f}',ha='center',va='bottom',fontsize=6.5)
    row_axes[0].set_ylabel('OOF AUC',fontsize=8)
    row_axes[0].text(-.18,1.12,letter,transform=row_axes[0].transAxes,fontsize=13,fontweight='bold',va='top')
draw_row(axs[0],fm,p_matrix(fm_p,'fm'),'A',fm_short)
draw_row(axs[1],rad,p_matrix(rad_p,'rad'),'B',rad_short)
fig.subplots_adjust(left=.12,right=.995,bottom=.15,top=.96,wspace=.18,hspace=.78)
for ext in ['pdf','png']:
    fig.savefig(OUT/f'Figure_S4_VOI_sensitivity_composite.{ext}',dpi=450 if ext=='png' else None,bbox_inches='tight',pad_inches=.03,facecolor='white')
plt.close(fig)
