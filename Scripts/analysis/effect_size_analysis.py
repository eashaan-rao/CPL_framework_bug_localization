"""
effect_size_analysis.py
────────────────────────
Computes effect sizes for CPL scenario comparisons.

Core question: are CPL gains large and losses small, or are they symmetric?
A 65% win rate where wins average +0.002 and losses average −0.15 does not
justify adopting CPL. This script provides the magnitude evidence.

Questions answered
──────────────────
1. What is Cohen's d for CP-transfer vs WP-small (and other comparisons)?
2. Is the gain distribution right-skewed (large wins, small losses)?
3. How do effect sizes differ between BLAZE and COOBA?
4. Does effect size vary by target codebase size group?

Outputs
───────
results/effect_size_summary.csv              — Cohen's d + Wilcoxon p per model × comparison × metric
results/effect_size_stratified_table.csv     — Mean metrics by model × scenario × target size group
results/images/es_delta_distributions.png
results/images/es_delta_by_target_size.png
results/images/es_cohens_d_heatmap.png

Run
───
    python Scripts/analysis/effect_size_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_CSV  = "/home/cs21d002_eashaan/PhD/Objective1/results/obj1_experimental_results.csv"
METADATA_PKL = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
IMG_DIR      = "/home/cs21d002_eashaan/PhD/Objective1/results/images"
OUT_CSV            = "/home/cs21d002_eashaan/PhD/Objective1/results/effect_size_summary.csv"
OUT_STRATIFIED_CSV = "/home/cs21d002_eashaan/PhD/Objective1/results/effect_size_stratified_table.csv"

os.makedirs(IMG_DIR, exist_ok=True)

COMPARISONS = [
    ('CP-transfer',   'WP-small',  'CPT vs WPS'),
    ('CP-transfer',   'WP-large',  'CPT vs WPL'),
    ('CP-cold-start', 'WP-small',  'CPC vs WPS'),
    ('WP-large',      'WP-small',  'WPL vs WPS'),
]
METRICS = ['MRR', 'MAP', 'top-1', 'top-5', 'top-10']

SIZE_BINS   = [0, 100_000, 300_000, np.inf]
SIZE_LABELS = ['Small\n(<100K)', 'Medium\n(100K–300K)', 'Large\n(>300K)']


def cohens_d_paired(diffs):
    """Cohen's d for paired differences: mean / std."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return np.nan
    return d.mean() / d.std(ddof=1)


def load_data():
    df   = pd.read_csv(RESULTS_CSV)
    meta = pd.read_parquet(METADATA_PKL)[['repo_name', 'LoC', 'total_unique_bug_reports']]
    df   = df.merge(
        meta.rename(columns={'repo_name': 'target_project', 'LoC': 'tgt_LoC',
                              'total_unique_bug_reports': 'tgt_n_bugs'}),
        on='target_project', how='left')
    return df


def compute_effect_sizes(df):
    rows = []
    for model, g in df.groupby('model_name'):
        for sc_a, sc_b, label in COMPARISONS:
            for metric in METRICS:
                a = (g[g['scenario'] == sc_a]
                     .set_index(['source_project', 'target_project'])[metric])
                b = (g[g['scenario'] == sc_b]
                     .set_index(['source_project', 'target_project'])[metric])
                common = a.index.intersection(b.index)
                if len(common) < 5:
                    continue
                diffs = (a[common] - b[common]).values
                d = cohens_d_paired(diffs)
                n_pos = int((diffs > 0).sum())
                n_neg = int((diffs < 0).sum())
                try:
                    _, p_gt = wilcoxon(diffs, alternative='greater', zero_method='zsplit')
                except ValueError:
                    p_gt = np.nan
                rows.append({
                    'model': model, 'comparison': label, 'metric': metric,
                    'n_pairs': len(common),
                    'mean_delta': float(np.nanmean(diffs)),
                    'median_delta': float(np.nanmedian(diffs)),
                    'std_delta': float(np.nanstd(diffs, ddof=1)),
                    'cohens_d': d,
                    'win_rate': n_pos / len(common),
                    'n_wins': n_pos,
                    'n_losses': n_neg,
                    'median_win':  float(np.median(diffs[diffs > 0])) if n_pos > 0 else np.nan,
                    'median_loss': float(np.median(diffs[diffs < 0])) if n_neg > 0 else np.nan,
                    'wilcoxon_p_gt': p_gt,
                })
    return pd.DataFrame(rows)


def build_delta_frame(df):
    rows = []
    for model, g in df.groupby('model_name'):
        cpt = g[g['scenario'] == 'CP-transfer'].set_index(['source_project', 'target_project'])
        wps = g[g['scenario'] == 'WP-small'].set_index(['source_project', 'target_project'])
        common = cpt.index.intersection(wps.index)
        for idx in common:
            rows.append({
                'model': model,
                'source_project': idx[0], 'target_project': idx[1],
                'delta_MRR': cpt.loc[idx, 'MRR'] - wps.loc[idx, 'MRR'],
                'tgt_LoC': cpt.loc[idx, 'tgt_LoC'],
                'tgt_n_bugs': cpt.loc[idx, 'tgt_n_bugs'],
            })
    delta = pd.DataFrame(rows)
    delta['tgt_size_group'] = pd.cut(
        delta['tgt_LoC'], bins=SIZE_BINS, labels=SIZE_LABELS)
    return delta


def plot_delta_distributions(delta):
    models = sorted(delta['model'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5))
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = delta[delta['model'] == model]['delta_MRR'].dropna()
        ax.hist(sub, bins=18, color='#2196F3', edgecolor='black', alpha=0.75)
        ax.axvline(0, color='red', linewidth=1.5, linestyle='--', label='Zero gain')
        ax.axvline(sub.mean(),   color='orange', linewidth=1.5, linestyle='-',
                   label=f'Mean = {sub.mean():.3f}')
        ax.axvline(sub.median(), color='green',  linewidth=1.5, linestyle='--',
                   label=f'Median = {sub.median():.3f}')
        ax.set_xlabel('CPL gain: CP-transfer − WP-small (MRR)', fontsize=10)
        ax.set_ylabel('Number of pairs', fontsize=10)
        ax.set_title(f'{model}: Distribution of CPL gains', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        n_pos = (sub > 0).sum();  n_neg = (sub < 0).sum()
        med_w = sub[sub > 0].median() if n_pos > 0 else 0.0
        med_l = sub[sub < 0].median() if n_neg > 0 else 0.0
        ax.text(0.98, 0.97,
                f'Wins: {n_pos}  median +{med_w:.3f}\nLosses: {n_neg}  median {med_l:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'es_delta_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ es_delta_distributions.png")


def plot_delta_by_target_size(delta):
    palette = {'BLAZE': '#2196F3', 'COOBA': '#FF9800'}
    fig, ax  = plt.subplots(figsize=(10, 6))
    sub = delta.dropna(subset=['tgt_size_group'])

    sns.violinplot(data=sub, x='tgt_size_group', y='delta_MRR', hue='model',
                   split=False, inner='box', palette=palette, ax=ax, alpha=0.8,
                   order=SIZE_LABELS)
    ax.axhline(0, color='red', linewidth=1.2, linestyle='--', alpha=0.8, label='Zero gain')
    ax.set_xlabel('Target codebase size', fontsize=11)
    ax.set_ylabel('CPL gain: CP-transfer − WP-small (MRR)', fontsize=11)
    ax.set_title('Effect size of CPL gain by target size group and model architecture',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'es_delta_by_target_size.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ es_delta_by_target_size.png")


def plot_cohens_d_heatmap(effect_df):
    sub = effect_df[effect_df['metric'] == 'MRR'].copy()
    sub['sig'] = sub['wilcoxon_p_gt'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))
    sub['label'] = sub['cohens_d'].apply(lambda d: f'{d:.2f}' if pd.notna(d) else 'N/A')
    sub['annot'] = sub['label'] + '\n' + sub['sig']

    models = sorted(sub['model'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    for ax, model in zip(axes, models):
        m = sub[sub['model'] == model].pivot(index='comparison', columns='metric', values='cohens_d')
        a = sub[sub['model'] == model].pivot(index='comparison', columns='metric', values='annot')
        col_order = [c for c in METRICS if c in m.columns]
        m = m[col_order]; a = a[col_order]
        sns.heatmap(m, annot=a, fmt='', cmap='RdYlGn', center=0,
                    vmin=-0.5, vmax=1.5, ax=ax, linewidths=0.5,
                    cbar_kws={'label': "Cohen's d"})
        ax.set_title(f"{model}: Cohen's d (MRR)\n* p<0.05  ** p<0.01  *** p<0.001",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Metric', fontsize=9)
        ax.set_ylabel('Comparison', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'es_cohens_d_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ es_cohens_d_heatmap.png")


def build_stratified_table(df):
    """Mean metrics per model × scenario × target LoC size group (Small/Medium/Large)."""
    SCENARIOS = ['WP-small', 'WP-large', 'CP-cold-start', 'CP-transfer']
    METRICS_ALL = ['MRR', 'MAP', 'top-1', 'top-5', 'top-10']

    df = df.copy()
    df['tgt_size_group'] = pd.cut(
        df['tgt_LoC'], bins=SIZE_BINS, labels=SIZE_LABELS)

    rows = []
    for model, mg in df.groupby('model_name'):
        for scenario in SCENARIOS:
            sg = mg[mg['scenario'] == scenario]
            # All targets combined
            n_pairs = len(sg)
            row = {'model': model, 'scenario': scenario, 'size_group': 'All', 'n_pairs': n_pairs}
            for m in METRICS_ALL:
                row[f'mean_{m}'] = sg[m].mean() if n_pairs > 0 else np.nan
            rows.append(row)
            # Per size group
            for size_label in SIZE_LABELS:
                sub = sg[sg['tgt_size_group'] == size_label]
                n = len(sub)
                row = {'model': model, 'scenario': scenario, 'size_group': size_label, 'n_pairs': n}
                for m in METRICS_ALL:
                    row[f'mean_{m}'] = sub[m].mean() if n > 0 else np.nan
                rows.append(row)

    return pd.DataFrame(rows)


def print_stratified_table(strat_df):
    print("\n── Stratified Results: Mean MRR by Model × Scenario × Target Size ───────")
    pd.set_option('display.float_format', '{:.3f}'.format)
    pivot = strat_df[strat_df['size_group'] != 'All'].pivot_table(
        index=['model', 'size_group'], columns='scenario',
        values='mean_MRR', aggfunc='first')
    scenario_order = ['WP-small', 'WP-large', 'CP-cold-start', 'CP-transfer']
    pivot = pivot[[c for c in scenario_order if c in pivot.columns]]
    print(pivot.to_string())
    print()
    print("── Pair counts by model × size group ───────────────────────────────────")
    counts = strat_df[strat_df['scenario'] == 'CP-transfer'].pivot_table(
        index='model', columns='size_group', values='n_pairs', aggfunc='first')
    print(counts.to_string())
    print()


def print_summary(effect_df):
    print("\n── Effect Size Summary (CP-transfer vs WP-small, MRR) ──────────────")
    sub = effect_df[
        (effect_df['comparison'] == 'CPT vs WPS') &
        (effect_df['metric'] == 'MRR')
    ]
    for _, row in sub.iterrows():
        sig = ('***' if row['wilcoxon_p_gt'] < 0.001 else
               ('**' if row['wilcoxon_p_gt'] < 0.01 else
                ('*' if row['wilcoxon_p_gt'] < 0.05 else 'ns')))
        med_l = f"{row['median_loss']:.3f}" if pd.notna(row['median_loss']) else 'N/A'
        print(f"  {row['model']}: Cohen's d = {row['cohens_d']:.3f}  "
              f"win_rate = {row['win_rate']:.1%}  "
              f"median_win = +{row['median_win']:.3f}  "
              f"median_loss = {med_l}  "
              f"Wilcoxon p = {row['wilcoxon_p_gt']:.4f} {sig}")
    print()


def main():
    print("Loading data...")
    df        = load_data()
    effect_df = compute_effect_sizes(df)
    delta_df  = build_delta_frame(df)

    print_summary(effect_df)
    plot_delta_distributions(delta_df)
    plot_delta_by_target_size(delta_df)
    plot_cohens_d_heatmap(effect_df)

    effect_df.to_csv(OUT_CSV, index=False)
    print(f"  ✓ {OUT_CSV}")

    strat_df = build_stratified_table(df)
    print_stratified_table(strat_df)
    strat_df.to_csv(OUT_STRATIFIED_CSV, index=False)
    print(f"  ✓ {OUT_STRATIFIED_CSV}")


if __name__ == '__main__':
    main()
