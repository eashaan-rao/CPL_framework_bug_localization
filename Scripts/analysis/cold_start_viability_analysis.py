"""
cold_start_viability_analysis.py
─────────────────────────────────
Identifies conditions under which CP-cold-start (zero target labels) achieves
acceptable bug localization performance.

Questions answered
──────────────────
1. Under what target LoC × source quality conditions is cold-start CPL viable?
2. Is there a target LoC threshold below which zero-shot CPL is practically useful?
3. How does BLAZE compare to COOBA in zero-shot transfer capability?

Outputs
───────
results/cold_start_viability.csv                    — per-pair CP-cold-start stats
results/images/cs_viability_heatmap_BLAZE.png
results/images/cs_viability_heatmap_COOBA.png
results/images/cs_loc_threshold.png

Run
───
    python Scripts/analysis/cold_start_viability_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_CSV  = "/home/cs21d002_eashaan/PhD/Objective1/results/paper_results_complete_corrected.csv"
METADATA_PKL = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
IMG_DIR      = "/home/cs21d002_eashaan/PhD/Objective1/results/images"
OUT_CSV      = "/home/cs21d002_eashaan/PhD/Objective1/results/cold_start_viability.csv"

os.makedirs(IMG_DIR, exist_ok=True)

VIABLE_MRR = 0.20  # "useful" cold-start threshold


def load_data():
    df   = pd.read_csv(RESULTS_CSV)
    meta = pd.read_parquet(METADATA_PKL)[['repo_name', 'LoC', 'total_unique_bug_reports']]
    df   = df.merge(
        meta.rename(columns={'repo_name': 'target_project', 'LoC': 'tgt_LoC',
                              'total_unique_bug_reports': 'tgt_n_bugs'}),
        on='target_project', how='left')
    df   = df.merge(
        meta.rename(columns={'repo_name': 'source_project', 'LoC': 'src_LoC',
                              'total_unique_bug_reports': 'src_n_bugs'}),
        on='source_project', how='left')
    return df


def build_viability_df(df):
    cpc = df[df['scenario'] == 'CP-cold-start'][
        ['model_name', 'source_project', 'target_project',
         'MRR', 'MAP', 'top-1', 'top-10',
         'tgt_LoC', 'tgt_n_bugs', 'src_LoC', 'src_n_bugs']
    ].copy()
    cpc['viable'] = cpc['MRR'] >= VIABLE_MRR

    # Attach WP-small MRR for comparison
    wps = df[df['scenario'] == 'WP-small'][
        ['model_name', 'source_project', 'target_project', 'MRR']
    ].rename(columns={'MRR': 'mrr_wps'})
    cpc = cpc.merge(wps, on=['model_name', 'source_project', 'target_project'], how='left')
    cpc['delta_vs_wps'] = cpc['MRR'] - cpc['mrr_wps']

    # Quartile bins — use all pairs pooled across models for consistent bin edges
    loc_bins  = pd.qcut(cpc['tgt_LoC'].dropna(), q=4, retbins=True)[1]
    bugs_bins = pd.qcut(cpc['src_n_bugs'].dropna(), q=4, retbins=True, duplicates='drop')[1]

    cpc['tgt_LoC_q'] = pd.cut(
        cpc['tgt_LoC'], bins=loc_bins, labels=['Q1\n(smallest)', 'Q2', 'Q3', 'Q4\n(largest)'],
        include_lowest=True)
    cpc['src_n_bugs_q'] = pd.cut(
        cpc['src_n_bugs'], bins=bugs_bins,
        labels=[f'Q{i+1}' for i in range(len(bugs_bins) - 1)],
        include_lowest=True)
    return cpc


def plot_heatmap(cpc, model):
    sub = cpc[cpc['model_name'] == model].dropna(subset=['tgt_LoC_q', 'src_n_bugs_q'])
    if sub.empty:
        print(f"  Skipping heatmap for {model}: no data")
        return

    pivot = sub.pivot_table(values='MRR', index='src_n_bugs_q', columns='tgt_LoC_q', aggfunc='mean', observed=False)
    count = sub.pivot_table(values='MRR', index='src_n_bugs_q', columns='tgt_LoC_q', aggfunc='count', observed=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.0, vmax=0.5, ax=ax, linewidths=0.5,
                cbar_kws={'label': f'Mean CP-cold-start MRR  (viable ≥ {VIABLE_MRR})'})

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            try:
                n = count.iloc[i, j]
                if pd.notna(n):
                    ax.text(j + 0.5, i + 0.82, f'n={int(n)}',
                            ha='center', va='center', fontsize=7, color='dimgrey')
            except (IndexError, ValueError):
                pass

    ax.set_xlabel('Target LoC quartile', fontsize=11)
    ax.set_ylabel('Source # bug reports quartile', fontsize=11)
    ax.set_title(
        f'{model}: CP-cold-start MRR by target size × source data richness\n'
        f'(green cells = viable, MRR ≥ {VIABLE_MRR})',
        fontsize=11, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(IMG_DIR, f'cs_viability_heatmap_{model}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ cs_viability_heatmap_{model}.png")


def plot_loc_threshold(cpc):
    models = sorted(cpc['model_name'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5))
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = cpc[cpc['model_name'] == model].dropna(subset=['tgt_LoC'])
        colors = sub['viable'].map({True: '#4CAF50', False: '#e57373'})
        ax.scatter(sub['tgt_LoC'] / 1000, sub['MRR'],
                   c=colors, alpha=0.78, s=65, edgecolors='k', linewidth=0.4, zorder=3)
        ax.axhline(VIABLE_MRR, color='darkorange', linewidth=1.3, linestyle='--',
                   label=f'Viable threshold (MRR = {VIABLE_MRR})', zorder=4)
        ax.set_xlabel('Target LoC (thousands)', fontsize=10)
        ax.set_ylabel('CP-cold-start MRR', fontsize=10)
        ax.set_title(f'{model}: Zero-shot CPL performance vs target codebase size',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        n_viable = sub['viable'].sum()
        loc_viable_max = sub.loc[sub['viable'], 'tgt_LoC'].max() if n_viable > 0 else np.nan
        info = (f"Viable: {n_viable}/{len(sub)} pairs"
                + (f"\nLargest viable tgt: {loc_viable_max/1000:.0f}K LoC" if pd.notna(loc_viable_max) else ''))
        ax.text(0.98, 0.97, info, transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'cs_loc_threshold.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ cs_loc_threshold.png")


def print_summary(cpc):
    print(f"\n── Cold-Start Viability (threshold: MRR ≥ {VIABLE_MRR}) ───────────────")
    for model, g in cpc.groupby('model_name'):
        n = len(g)
        n_viable = g['viable'].sum()
        print(f"  {model}: viable={n_viable}/{n} ({100*n_viable/n:.0f}%)")
        print(f"    MRR: mean={g['MRR'].mean():.3f}  median={g['MRR'].median():.3f}  "
              f"max={g['MRR'].max():.3f}")
        v  = g[g['viable']]['tgt_LoC']
        nv = g[~g['viable']]['tgt_LoC']
        if len(v) > 0 and len(nv) > 0:
            print(f"    Median tgt_LoC — viable: {v.median()/1000:.0f}K  "
                  f"not viable: {nv.median()/1000:.0f}K")
        print()


def main():
    print("Loading data...")
    df  = load_data()
    cpc = build_viability_df(df)

    print_summary(cpc)
    for model in sorted(cpc['model_name'].unique()):
        plot_heatmap(cpc, model)
    plot_loc_threshold(cpc)

    cpc.to_csv(OUT_CSV, index=False)
    print(f"  ✓ {OUT_CSV}")


if __name__ == '__main__':
    main()
