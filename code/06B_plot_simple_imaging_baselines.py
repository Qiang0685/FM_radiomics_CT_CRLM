#!/usr/bin/env python3
"""Create the supplementary simple-imaging-baseline AUC figure.

The final analysis includes mean liver parenchyma HU for steatosis and spleen
volume for sinusoidal dilatation.
"""
from pathlib import Path
import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd

MODELS = ['Merlin', 'RADAR', 'TotalFM', 'Jolia', 'OrganLens', 'Radiomics']
COLORS = {'Merlin':'#4C78A8','RADAR':'#59A14F','TotalFM':'#F28E2B',
          'Jolia':'#B279A2','OrganLens':'#76B7B2','Radiomics':'#7F7F7F'}
BASELINE_COLORS = {'Mean liver parenchyma HU':'#8C8C8C', 'Spleen volume':'#8C8C8C'}
BASELINE_BY_TASK = {'Steatosis': 'Mean liver parenchyma HU',
                    'Sinusoidal dilatation': 'Spleen volume'}

def unpack(value):
    m = re.match(r'\s*([0-9.]+)\s*\(([0-9.]+)-([0-9.]+)\)', str(value))
    if not m:
        raise ValueError(f'Cannot parse AUC CI: {value!r}')
    return tuple(float(x) for x in m.groups())

def main():
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--table',type=Path,required=True); parser.add_argument('--out-dir',type=Path,required=True); args=parser.parse_args()
 df = pd.read_csv(args.table)
 if 'AUC (95% CI)' not in df:
     df['AUC (95% CI)'] = df.apply(lambda r: f"{r.AUC:.3f} ({r.CI_low:.3f}-{r.CI_high:.3f})", axis=1)
 tasks = ['Steatosis', 'Sinusoidal dilatation']
 fig, axes = plt.subplots(1, 2, figsize=(180/25.4, 78/25.4), sharey=True)
 for ax, task in zip(axes, tasks):
     sub = df[df.Task.eq(task)].copy()
     baseline = BASELINE_BY_TASK[task]
     names = MODELS + [baseline]
     missing = set(names).difference(sub.Predictor)
     extras = set(sub.Predictor).difference(names)
     if missing or extras:
         raise ValueError(f'Unexpected predictors for {task}: missing={sorted(missing)}, extras={sorted(extras)}')
     for i, name in enumerate(names):
         row = sub[sub.Predictor.eq(name)].iloc[0]
         auc, lo, hi = unpack(row['AUC (95% CI)'])
         color = COLORS.get(name, BASELINE_COLORS.get(name, '#8C8C8C'))
         ax.bar(i, auc, color=color, alpha=0.86, width=0.66, edgecolor='#333333', linewidth=0.45)
         ax.errorbar(i, auc, yerr=[[auc-lo], [hi-auc]], color='#30343B', capsize=2.4,
                     linewidth=1.0, capthick=1.0, zorder=3)
     ax.axhline(0.5, color='#777777', linestyle='--', linewidth=.75)
     ax.set_xticks(range(len(names)), ['Liver HU' if n == 'Mean liver parenchyma HU' else
                                       'Spleen volume' if n == 'Spleen volume' else n for n in names],
                   rotation=35, ha='right', fontsize=6.2)
     ax.set_ylim(0, 1.0)
     ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.tick_params(axis='y', labelsize=6.2)
     ax.grid(axis='y', color='#E1E4E8', linewidth=.45); ax.set_axisbelow(True)
     ax.spines[['top','right']].set_visible(False)
     ax.set_title(task, fontsize=8.5, pad=4)
 axes[0].set_ylabel('OOF AUC', fontsize=7.5)
 fig.subplots_adjust(left=.065, right=.995, bottom=.27, top=.88, wspace=.16)
 args.out_dir.mkdir(parents=True, exist_ok=True)
 fig.savefig(args.out_dir/'Figure_S3_FM_radiomics_imaging_baselines.pdf', facecolor='white')
 fig.savefig(args.out_dir/'Figure_S3_FM_radiomics_imaging_baselines.png', dpi=400, facecolor='white')
 plt.close(fig)

if __name__ == '__main__': main()
