#!/usr/bin/env python3
"""Render Figure 5 using the established archived layout and styling."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

FMS = ['Merlin', 'RADAR', 'TotalFM', 'Jolia', 'OrganLens']
TASKS = ['Steatosis', 'NAS ≥ 3', 'Sinusoidal dilatation']
TASK_SLUG = {'Steatosis':'steatosis','NAS ≥ 3':'nas3','Sinusoidal dilatation':'sinusoidal_dilatation'}
COLORS = {'Merlin':'#4C78A8','RADAR':'#59A14F','TotalFM':'#F28E2B','Jolia':'#B279A2','OrganLens':'#76B7B2'}


def pair_value(frame, task, model, column):
    key = 'NAS >= 3' if task == 'NAS ≥ 3' else task
    d = frame[frame.Task.eq(key)]
    row = d[(d.Model_1.eq(model)) & (d.Model_2.eq('Radiomics'))]
    if row.empty:
        row = d[(d.Model_2.eq(model)) & (d.Model_1.eq('Radiomics'))]
    return float(row.iloc[0][column])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fusion-dir', type=Path, required=True)
    ap.add_argument('--overlap', type=Path, required=True)
    ap.add_argument('--oof-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    pair = pd.read_csv(a.fusion_dir / 'pairwise_complementarity_raw.csv')
    overlap = pd.read_csv(a.overlap)
    oof = {}
    for task in TASKS:
        oof[task] = pd.read_csv(a.oof_dir / f'oof_{TASK_SLUG[task]}.csv')

    # Exact archived geometry: 3 stacked strips, then 3 aligned fusion panels.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':1.2,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig = plt.figure(figsize=(7.29,7.30), facecolor='white')
    outer = fig.add_gridspec(3,1,height_ratios=[2.5,2.5,5],hspace=.38,left=.18,right=.965,top=.965,bottom=.07)
    rho_cmap = LinearSegmentedColormap.from_list('rho',['#EAF3F8','#4E9AC6','#064B91'])
    jac_cmap = LinearSegmentedColormap.from_list('jac',['#F4F3FA','#AAA7CE','#4B0082'])

    def draw_strip_panel(slot, source, value_col, cmap, vmin, vmax, label, letter, with_p=False):
        subgrid = slot.subgridspec(1,2,width_ratios=[5.6,.18],wspace=.20)
        rows = subgrid[0,0].subgridspec(3,1,hspace=.08)
        norm = Normalize(vmin=vmin,vmax=vmax)
        for i,task in enumerate(TASKS):
            ax=fig.add_subplot(rows[i,0])
            vals = [pair_value(source,task,fm,value_col) for fm in FMS]
            arr=np.asarray(vals,float)[None,:]
            ax.imshow(arr,cmap=cmap,norm=norm,aspect='auto',interpolation='none')
            ax.set_yticks([]); ax.set_xticks(np.arange(len(FMS)))
            ax.set_xticklabels(FMS if i==2 else [],rotation=35 if i==2 else 0,
                               ha='right' if i==2 else 'center',fontsize=9)
            for j,v in enumerate(vals):
                text=f'{v:.3f}'
                if with_p:
                    p=spearmanr(oof[task][FMS[j]],oof[task]['Radiomics']).pvalue
                    text += f'\n{"p<.001" if p<.001 else f"p={p:.3f}"}'
                ax.text(j,0,text,ha='center',va='center',fontsize=8.5,color='#202020',linespacing=1.2)
            ax.text(-.02,.5,task.replace('Sinusoidal dilatation','Sinusoidal\ndilatation'),transform=ax.transAxes,
                    ha='right',va='center',fontsize=9.5)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.tick_params(length=0)
        cax=fig.add_subplot(subgrid[0,1]); sm=ScalarMappable(norm=norm,cmap=cmap); sm.set_array([])
        cb=fig.colorbar(sm,cax=cax); cb.set_ticks([vmin,(vmin+vmax)/2,vmax]); cb.ax.tick_params(labelsize=8,width=.8,pad=2); cb.outline.set_linewidth(.5)
        cax.set_title(label,fontsize=9.5,pad=5)
        fig.text(.075,.985 if letter=='A' else .720,letter,fontsize=13,fontweight='bold',va='top',ha='left')

    draw_strip_panel(outer[0],pair,'Spearman_rho_OOF_probability',rho_cmap,0,.75,'Spearman ρ','A',True)
    draw_strip_panel(outer[1],overlap,'Error_Jaccard',jac_cmap,.15,.55,'Error Jaccard','B',False)

    bot=outer[2].subgridspec(2,1,height_ratios=[.30,1],hspace=.02)
    axs=[fig.add_subplot(bot[1,0].subgridspec(1,3,wspace=.25)[0,i]) for i in range(3)]
    for ax,task in zip(axs,TASKS):
        d=oof[task]; x=np.arange(len(FMS)); fm_auc=[roc_auc_score(d.y_true,d[fm]) for fm in FMS]
        fusion_auc=[pair_value(pair,task,fm,'Equal_pair_AUC') for fm in FMS]
        rad_auc=roc_auc_score(d.y_true,d.Radiomics)
        for i,fm in enumerate(FMS):
            col=COLORS[fm]
            ax.plot([i,i],[fm_auc[i],fusion_auc[i]],color=col,lw=1.8,zorder=2)
            ax.scatter(i,fm_auc[i],s=56,facecolor='white',edgecolor=col,linewidth=1.5,zorder=3,label='FM' if i==0 else None)
            ax.scatter(i,fusion_auc[i],s=68,marker='D',facecolor=col,edgecolor=col,linewidth=.7,zorder=4,label='Radiomics + FM' if i==0 else None)
        ax.axhline(rad_auc,color='#7F7F7F',linestyle=(0,(1.2,2.4)),lw=1.9,label='Radiomics' if task==TASKS[0] else None)
        ax.axhline(.5,color='#888',linestyle=(0,(3,2)),lw=1.2)
        ax.set_title(task if task!='Sinusoidal dilatation' else 'Sinusoidal\ndilatation',fontsize=9.5,pad=4)
        ax.set_xticks(x); ax.set_xticklabels(FMS,rotation=35,ha='right',fontsize=9)
        ax.set_ylim(.4,.92); ax.set_yticks([.4,.6,.8,.9]); ax.set_ylabel('OOF AUC',fontsize=9.5) if ax is axs[0] else ax.set_yticklabels([])
        ax.grid(axis='y',color='#D9DDE2',linewidth=.8); ax.set_axisbelow(True); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(width=1.1)
    fig.text(.075,.345,'C',fontsize=13,fontweight='bold',va='top',ha='left')
    handles,labels=axs[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.59,.440),ncol=3,frameon=False,fontsize=9,handletextpad=.5,columnspacing=1.0)
    for ext in ['pdf','png']:
        fig.savefig(a.out_dir/f'Figure_5_prediction_complementarity_fusion.{ext}',dpi=450 if ext=='png' else None,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)


if __name__=='__main__': main()
