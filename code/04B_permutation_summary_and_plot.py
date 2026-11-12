#!/usr/bin/env python3
"""Validate, summarize, and plot all 18 model/task permutation cells."""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TASKS = [('Steatosis', 'steatosis'), ('NAS >= 3', 'nas3'),
         ('Sinusoidal dilatation', 'sinusoidal_dilatation')]
MODELS = ['Merlin', 'RADAR', 'TotalFM', 'Jolia', 'OrganLens', 'Radiomics']
COLORS = {'Merlin':'#4C78A8','RADAR':'#59A14F','TotalFM':'#F28E2B',
          'Jolia':'#B279A2','OrganLens':'#76B7B2','Radiomics':'#7F7F7F'}

def auc_u2(y, score):
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    pos = score[y == 1]; neg = score[y == 0]
    return int(2 * np.sum(pos[:, None] > neg[None, :]) +
               np.sum(pos[:, None] == neg[None, :]))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty(len(p)); out[order] = np.minimum(adj, 1)
    return out


def merge_cells(cell_dir: Path, oof_dir: Path, output_dir: Path, n_permutations: int):
    rows, hashes = [], {}
    for task, slug in TASKS:
        path = oof_dir / f'main_oof_{slug}.csv'
        if not path.exists(): path = oof_dir / f'oof_{slug}.csv'
        oof = pd.read_csv(path); hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        y = oof.y_true.to_numpy(int)
        for model in MODELS:
            cell = cell_dir / f'{slug}__{model}'
            meta = cell / f'{slug}__{model}_result.json'
            null_path = cell / f'{slug}__{model}_null_auc.csv'
            if not meta.exists() or not null_path.exists(): raise FileNotFoundError(cell)
            meta_data = json.loads(meta.read_text()); null = pd.read_csv(null_path)
            if len(null) != n_permutations or not np.array_equal(null.permutation.to_numpy(), np.arange(1, n_permutations + 1)):
                raise ValueError(f'Incorrect permutation count/index for {cell}')
            observed = roc_auc_score(y, oof[model].to_numpy(float))
            if not np.isclose(meta_data['Observed_OOF_AUC'], observed, atol=1e-12): raise ValueError(f'Observed AUC mismatch for {cell}')
            null_auc = null.null_auc.to_numpy(float)
            observed_u2 = auc_u2(y, oof[model].to_numpy(float))
            if 'null_u2' in null.columns:
                null_u2 = null.null_u2.to_numpy(int)
            else:
                # Backward-compatible exact reconstruction for old checkpoints.
                null_u2 = np.rint(2 * null_auc * int(y.sum()) * int((y == 0).sum())).astype(int)
            exceed = int(np.sum(null_u2 >= observed_u2))
            p = (exceed + 1) / (n_permutations + 1)
            if exceed != meta_data['Exceedances'] or not np.isclose(p, meta_data['Empirical_upper_tail_p'], atol=1e-12): raise ValueError(f'Empirical p mismatch for {cell}')
            rows.append({'Task': task, 'Model': model, 'N': len(y), 'Events': int(y.sum()),
                         'Observed_OOF_AUC': observed, 'N_permutations': n_permutations,
                         'Null_AUC_mean': float(null_auc.mean()), 'Null_AUC_SD': float(null_auc.std(ddof=1)),
                         'Null_AUC_95th_percentile': float(np.quantile(null_auc, .95)), 'Exceedances': exceed,
                         'Empirical_upper_tail_p': p,
                         'Monte_Carlo_exceedance_CI_low': meta_data['Exceedance_probability_CI_low'],
                         'Monte_Carlo_exceedance_CI_high': meta_data['Exceedance_probability_CI_high']})
    out = pd.DataFrame(rows); out['Holm_p_within_task'] = np.nan
    for task, _ in TASKS:
        idx = out.index[out.Task == task]; out.loc[idx, 'Holm_p_within_task'] = holm(out.loc[idx, 'Empirical_upper_tail_p'])
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / 'permutation_18cells_1000_raw.csv', index=False)
    manifest = {'analysis': 'fixed seed-averaged OOF label permutation', 'model_order': MODELS,
                'task_order': [t for t, _ in TASKS], 'n_permutations_per_cell': n_permutations,
                'total_cells': len(rows), 'outer_seeds': [42, 0, 1, 7, 2024], 'outer_folds': 5,
                'inner_folds': 3, 'C_grid': [.01, .1, 1., 10.],
                'p_formula': f'(1 + count(null_AUC >= observed_AUC))/({n_permutations} + 1)',
                'multiple_comparison_summaries': 'Holm adjustment within each task across the six representations',
                'source_oof_sha256': hashes, 'python_for_merge': platform.python_version()}
    (output_dir / 'permutation_run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return out


def plot_permutation(summary: pd.DataFrame, cell_dir: Path, figure_dir: Path, n_permutations: int):
    import matplotlib.pyplot as plt
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(180/25.4, 76/25.4), sharey=True)
    for ax, (title, slug) in zip(axes, TASKS):
        subset = summary[summary.Task == title]
        vals, observed = [], []
        for model in MODELS:
            row = subset[subset.Model == model].iloc[0]
            null = pd.read_csv(cell_dir / f'{slug}__{model}' / f'{slug}__{model}_null_auc.csv')['null_auc'].dropna().to_numpy(float)
            if len(null) != n_permutations: raise ValueError(f'Unexpected null count for {slug}/{model}')
            vals.append(null); observed.append(float(row.Observed_OOF_AUC))
        bp = ax.boxplot(vals, positions=np.arange(1, 7), widths=.58, patch_artist=True, showfliers=False,
                        medianprops={'color':'#242A31','linewidth':1.1}, whiskerprops={'color':'#5B616A','linewidth':.8},
                        capprops={'color':'#5B616A','linewidth':.8}, boxprops={'edgecolor':'#5B616A','linewidth':.7})
        for box, model in zip(bp['boxes'], MODELS): box.set_facecolor(COLORS[model]); box.set_alpha(.58)
        for i, model in enumerate(MODELS, 1): ax.scatter(i, observed[i-1], color=COLORS[model], edgecolor='#222222', linewidth=.55, s=19, zorder=4)
        ax.axhline(.5, color='#777777', linestyle='--', linewidth=.8)
        ax.set_title(title, fontsize=8); ax.set_xticks(range(1, 7), MODELS, rotation=35, ha='right', fontsize=6)
        ax.set_ylim(0, .98); ax.set_yticks(np.arange(0, 1.0, .1)); ax.tick_params(axis='y', labelsize=6)
        ax.grid(axis='y', color='#E1E4E8', linewidth=.45); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('OOF AUC', fontsize=7); fig.subplots_adjust(left=.065, right=.995, bottom=.24, top=.90, wspace=.17)
    fig.savefig(figure_dir / 'Figure_S1_permutation_tests_18_cells.pdf', facecolor='white')
    fig.savefig(figure_dir / 'Figure_S1_permutation_tests_18_cells.png', dpi=400, facecolor='white'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell-dir', type=Path, required=True); ap.add_argument('--oof-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True); ap.add_argument('--figure-dir', type=Path)
    ap.add_argument('--n-permutations', type=int, default=1000); a = ap.parse_args()
    summary = merge_cells(a.cell_dir, a.oof_dir, a.output_dir, a.n_permutations)
    plot_permutation(summary, a.cell_dir, a.figure_dir or a.output_dir, a.n_permutations)
    print('Validated, merged, and plotted', len(summary), 'cells')


if __name__ == '__main__': main()
