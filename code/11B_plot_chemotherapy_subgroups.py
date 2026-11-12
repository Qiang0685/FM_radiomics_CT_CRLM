from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser(); ap.add_argument('--subgroup',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); args=ap.parse_args()
OUT=args.out_dir; OUT.mkdir(parents=True,exist_ok=True)
data=pd.read_csv(args.subgroup)
tasks=['Steatosis','NAS ≥ 3','Sinusoidal dilatation']
data['Task']=data['Task'].replace({'NAS >= 3':'NAS ≥ 3'})
models=['Merlin','RADAR','TotalFM','Jolia','OrganLens','Radiomics']
colors={'Chemotherapy':'#4C78A8','No chemotherapy':'#7F7F7F'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(3,1,figsize=(7.09,5.2),sharex=True)
x=np.arange(len(models)); width=.30
for idx,(ax,task) in enumerate(zip(axes,tasks)):
    d=data[data.Task.eq(task)]
    for status,offset in [('Chemotherapy',-.16),('No chemotherapy',.16)]:
        s=d[d.Chemo_status.eq(status)].set_index('Model').reindex(models)
        vals=s.AUC.to_numpy(float); lo=s.CI_low.to_numpy(float); hi=s.CI_high.to_numpy(float)
        ax.bar(x+offset,vals,width,color=colors[status],label=status if idx==0 else None,
               yerr=np.vstack([vals-lo,hi-vals]),error_kw={'ecolor':'#30343B','elinewidth':.9,'capsize':2.5,'capthick':.9})
    ax.axhline(.5,color='#777',linestyle='--',linewidth=.8)
    ax.set_ylim(0,1.02); ax.set_yticks(np.arange(0,1.01,.2)); ax.set_ylabel('OOF AUC',fontsize=8)
    ax.set_title(task,loc='left',fontsize=9.5,fontweight='bold',pad=5)
    ax.grid(axis='y',color='#E1E4E8',linewidth=.5); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False); ax.tick_params(axis='y',labelsize=7)
axes[-1].set_xticks(x,models,fontsize=8)
axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.40),ncol=2,frameon=False,fontsize=8.5)
fig.subplots_adjust(left=.07,right=.995,bottom=.16,top=.92,hspace=.42)
for ext in ['pdf','png']:
    fig.savefig(OUT/f'Figure_S6_subgroup_analysis.{ext}',dpi=450 if ext=='png' else None,bbox_inches='tight',pad_inches=.03,facecolor='white')
plt.close(fig)
