#!/usr/bin/env python3
"""Run fixed-OOF label permutations for all 18 task/model cells."""
from __future__ import annotations
import argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

TASKS = [('Steatosis', 'steatosis'), ('NAS >= 3', 'nas3'), ('Sinusoidal dilatation', 'sinusoidal_dilatation')]
MODELS = ['Merlin', 'RADAR', 'TotalFM', 'Jolia', 'OrganLens', 'Radiomics']

def auc_u2(y, score):
    """Return 2U, an integer tie-aware Mann–Whitney statistic."""
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    pos = score[y == 1]; neg = score[y == 0]
    # Exact comparison avoids floating-point undercounting of equal AUCs.
    return int(2 * np.sum(pos[:, None] > neg[None, :]) +
               np.sum(pos[:, None] == neg[None, :]))


def run_cell(task_slug, model, args):
    folder = args.output_dir / f'{task_slug}__{model}'; folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / f'{task_slug}__{model}_null_auc.csv'; meta = folder / f'{task_slug}__{model}_result.json'
    f = args.oof_dir / f'main_oof_{task_slug}.csv'
    if not f.exists(): f = args.oof_dir / f'oof_{task_slug}.csv'
    oof = pd.read_csv(f, dtype={'case_id': str})
    y = oof.y_true.to_numpy(int)
    score = oof[model].to_numpy(float)
    observed = roc_auc_score(y, score); observed_u2 = auc_u2(y, score)
    null = pd.read_csv(checkpoint).null_auc.tolist() if checkpoint.exists() else []
    if len(null) > args.n_permutations: raise ValueError(f'Checkpoint exceeds requested permutations: {checkpoint}')
    # Preserve the original cell-wise reproducibility: each independently
    # launched cell used the same explicit seed and therefore the same
    # permutation-label sequence.
    rng = np.random.default_rng(args.seed)
    for _ in range(len(null)): rng.permutation(y)
    null_u2 = []
    while len(null) < args.n_permutations:
        yp = rng.permutation(y)
        null.append(roc_auc_score(yp, score))
        null_u2.append(auc_u2(yp, score))
        pd.DataFrame({'permutation': np.arange(1, len(null) + 1), 'null_auc': null,
                      'null_u2': null_u2}).to_csv(checkpoint, index=False)
    # Recompute the integer statistic from the complete checkpoint so resumed
    # runs are exact even when the checkpoint predates the null_u2 column.
    if checkpoint.exists():
        ck = pd.read_csv(checkpoint)
        if 'null_u2' in ck.columns and len(ck) == len(null):
            null_u2 = ck.null_u2.astype(int).tolist()
        else:
            rng2 = np.random.default_rng(args.seed)
            null_u2 = []
            for _ in range(len(null)):
                null_u2.append(auc_u2(rng2.permutation(y), score))
    exceed = int(np.sum(np.asarray(null_u2, dtype=int) >= observed_u2)); p = (exceed + 1) / (args.n_permutations + 1)
    ci = binomtest(exceed, args.n_permutations).proportion_ci(confidence_level=.95, method='exact')
    result = {'Task': task_slug, 'Model': model, 'Observed_OOF_AUC': observed, 'N_permutations': args.n_permutations,
              'Null_AUC_mean': float(np.mean(null)), 'Null_AUC_SD': float(np.std(null, ddof=1)),
              'Null_AUC_95th_percentile': float(np.quantile(null, .95)), 'Observed_U2': observed_u2,
              'Exceedances': exceed,
              'Empirical_upper_tail_p': p, 'Exceedance_probability_CI_low': float(ci.low),
              'Exceedance_probability_CI_high': float(ci.high),
              'Interpretation': 'If observed AUC <0.5, do not interpret as reverse biological signal.'}
    meta.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--oof-dir', type=Path, required=True); ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--tasks', nargs='+', choices=[s for _, s in TASKS], default=[s for _, s in TASKS]); ap.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    ap.add_argument('--n-permutations', type=int, default=1000); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--seed', type=int, default=20261103)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = [(t, m) for t in args.tasks for m in args.models]; results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_cell, t, m, args) for t, m in cells]
        for future in as_completed(futures): results.append(future.result())
    order = {(t, m): i for i, (t, m) in enumerate(cells)}
    results.sort(key=lambda r: order[(r['Task'], r['Model'])])
    pd.DataFrame(results).to_csv(args.output_dir/'permutation_summary_completed_cells.csv', index=False)
    if len(results) != len(cells): raise SystemExit(f'Only {len(results)}/{len(cells)} cells completed')
    print(f'All {len(cells)} permutation cells complete')


if __name__ == '__main__': main()
