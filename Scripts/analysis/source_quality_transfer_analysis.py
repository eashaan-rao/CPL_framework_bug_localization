"""
source_quality_transfer_analysis.py
─────────────────────────────────────
Tests whether a source project's within-project model quality (WP-large MRR
when that project is itself a target) predicts how well it transfers to a
different target in the CPL setting.

Definition: "source quality" for project S = mean WP-large MRR across all
rows where S is the *target* project. This reflects how learnable S's codebase
is — a proxy for how rich and reliable a training signal it provides.

Questions answered
──────────────────
1. Does high source quality → higher CP-transfer MRR on the actual target?
2. Does high source quality → positive CPL gain (vs low quality → negative transfer)?
3. Is this effect consistent across BLAZE and COOBA?

Outputs
───────
results/source_quality_transfer.csv        — per-pair source quality + CPL metrics
results/images/sq_scatter_BLAZE.png
results/images/sq_scatter_COOBA.png

Run
───
    python Scripts/analysis/source_quality_transfer_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
from scipy.stats import spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_CSV  = "/home/cs21d002_eashaan/PhD/Objective1/results/paper_results_complete_corrected.csv"
METADATA_PKL = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
IMG_DIR      = "/home/cs21d002_eashaan/PhD/Objective1/results/images"
OUT_CSV      = "/home/cs21d002_eashaan/PhD/Objective1/results/source_quality_transfer.csv"

os.makedirs(IMG_DIR, exist_ok=True)

NEG_THRESH = -0.01
POS_THRESH =  0.01


def load_data():
    df   = pd.read_csv(RESULTS_CSV)
    meta = pd.read_parquet(METADATA_PKL)[['repo_name', 'LoC', 'total_unique_bug_reports']]
    df   = df.merge(
        meta.rename(columns={'repo_name': 'target_project', 'LoC': 'tgt_LoC',
                              'total_unique_bug_reports': 'tgt_n_bugs'}),
        on='target_project', how='left')
    return df


def build_analysis_df(df):
    rows = []
    for model, g in df.groupby('model_name'):
        # Source quality: mean WP-large MRR when each project appears as a target
        src_quality = (g[g['scenario'] == 'WP-large']
                       .groupby('target_project')['MRR']
                       .mean()
                       .rename('src_wpl_mrr'))

        cpt = g[g['scenario'] == 'CP-transfer'].set_index(['source_project', 'target_project'])
        wps = g[g['scenario'] == 'WP-small'].set_index(['source_project', 'target_project'])['MRR']
        wpl = g[g['scenario'] == 'WP-large'].set_index(['source_project', 'target_project'])['MRR']

        for (src, tgt), row in cpt.iterrows():
            sq = src_quality.get(src, np.nan)
            wps_val = wps.get((src, tgt), np.nan)
            rows.append({
                'model': model,
                'source_project': src, 'target_project': tgt,
                'src_wpl_mrr': sq,
                'cpt_mrr': row['MRR'],
                'wps_mrr': wps_val,
                'wpl_mrr': wpl.get((src, tgt), np.nan),
                'delta_mrr': row['MRR'] - wps_val if not pd.isna(wps_val) else np.nan,
                'tgt_LoC': row.get('tgt_LoC', np.nan),
                'tgt_n_bugs': row.get('tgt_n_bugs', np.nan),
            })
    analysis = pd.DataFrame(rows)
    analysis['transfer'] = analysis['delta_mrr'].apply(
        lambda d: 'positive' if d > POS_THRESH else ('negative' if d < NEG_THRESH else 'neutral')
        if not pd.isna(d) else 'unknown')
    return analysis


def plot_scatter(analysis, model):
    sub = analysis[analysis['model'] == model].dropna(subset=['src_wpl_mrr', 'cpt_mrr'])
    if len(sub) < 3:
        print(f"  Skipping {model}: insufficient data")
        return

    color_map = {'positive': '#4CAF50', 'neutral': '#FFC107',
                 'negative': '#F44336', 'unknown': '#9E9E9E'}
    colors = sub['transfer'].map(color_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: source quality vs CP-transfer MRR ──────────────────────────────
    ax = axes[0]
    ax.scatter(sub['src_wpl_mrr'], sub['cpt_mrr'],
               c=colors, s=70, edgecolors='k', linewidth=0.4, alpha=0.85, zorder=3)

    mask = sub['src_wpl_mrr'].notna() & sub['cpt_mrr'].notna()
    if mask.sum() > 2:
        slope, intercept, r, p_r, _ = scipy_stats.linregress(
            sub.loc[mask, 'src_wpl_mrr'], sub.loc[mask, 'cpt_mrr'])
        x_line = np.linspace(sub['src_wpl_mrr'].min(), sub['src_wpl_mrr'].max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                'b--', linewidth=1.5, alpha=0.7, label=f'OLS  r={r:.2f}')

    rho, p = spearmanr(sub['src_wpl_mrr'], sub['cpt_mrr'])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

    for _, row in sub.iterrows():
        ax.annotate(row['source_project'].split('/')[-1],
                    (row['src_wpl_mrr'], row['cpt_mrr']),
                    fontsize=6, ha='left', va='bottom', alpha=0.65)

    ax.set_xlabel('Source quality: WP-large MRR\n(source project as its own target)',
                  fontsize=10)
    ax.set_ylabel('CP-transfer MRR (on actual target)', fontsize=10)
    ax.set_title(f'{model}: Source quality → transfer quality', fontsize=11, fontweight='bold')
    ax.text(0.03, 0.97, f'Spearman ρ = {rho:.2f} {sig}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # ── Right: source quality vs CPL gain ────────────────────────────────────
    ax = axes[1]
    sub2 = sub.dropna(subset=['src_wpl_mrr', 'delta_mrr'])
    colors2 = sub2['transfer'].map(color_map)
    ax.scatter(sub2['src_wpl_mrr'], sub2['delta_mrr'],
               c=colors2, s=70, edgecolors='k', linewidth=0.4, alpha=0.85, zorder=3)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)

    rho2, p2 = spearmanr(sub2['src_wpl_mrr'], sub2['delta_mrr'])
    sig2 = '***' if p2 < 0.001 else ('**' if p2 < 0.01 else ('*' if p2 < 0.05 else 'ns'))

    ax.set_xlabel('Source quality: WP-large MRR', fontsize=10)
    ax.set_ylabel('CPL gain: CP-transfer − WP-small (MRR)', fontsize=10)
    ax.set_title(f'{model}: Source quality → CPL gain', fontsize=11, fontweight='bold')
    ax.text(0.03, 0.97, f'Spearman ρ = {rho2:.2f} {sig2}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.grid(alpha=0.3)

    handles = [
        mpatches.Patch(color='#4CAF50', label='Positive transfer (>+0.01)'),
        mpatches.Patch(color='#FFC107', label='Neutral (±0.01)'),
        mpatches.Patch(color='#F44336', label='Negative transfer (<−0.01)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    out = os.path.join(IMG_DIR, f'sq_scatter_{model}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ sq_scatter_{model}.png")


def print_summary(analysis):
    print("\n── Source Quality → Transfer Quality ──────────────────────────────────")
    for model, g in analysis.groupby('model'):
        sub = g.dropna(subset=['src_wpl_mrr', 'cpt_mrr'])
        if len(sub) < 3:
            continue
        rho1, p1 = spearmanr(sub['src_wpl_mrr'], sub['cpt_mrr'])
        sig1 = '***' if p1 < 0.001 else ('**' if p1 < 0.01 else ('*' if p1 < 0.05 else 'ns'))

        sub2 = g.dropna(subset=['src_wpl_mrr', 'delta_mrr'])
        rho2 = p2 = np.nan
        if len(sub2) >= 3:
            rho2, p2 = spearmanr(sub2['src_wpl_mrr'], sub2['delta_mrr'])
        sig2 = '***' if (p2 < 0.001) else ('**' if (p2 < 0.01) else ('*' if (p2 < 0.05) else 'ns')) if pd.notna(p2) else ''

        print(f"  {model}:")
        print(f"    source_quality → cpt_mrr:   ρ = {rho1:+.3f}  p = {p1:.4f}  {sig1}")
        print(f"    source_quality → delta_mrr: ρ = {rho2:+.3f}  p = {p2:.4f}  {sig2}" if pd.notna(rho2) else
              f"    source_quality → delta_mrr: insufficient data")

        # Top 3 sources by quality and their transfer stats
        top3 = sub2.sort_values('src_wpl_mrr', ascending=False).drop_duplicates('source_project').head(3)
        bot3 = sub2.sort_values('src_wpl_mrr').drop_duplicates('source_project').head(3)
        print(f"    Top-3 sources by quality: mean delta = "
              f"{top3['delta_mrr'].mean():.3f} | "
              f"Bottom-3: {bot3['delta_mrr'].mean():.3f}")
        print()


def main():
    print("Loading data...")
    df       = load_data()
    analysis = build_analysis_df(df)

    print_summary(analysis)
    for model in sorted(analysis['model'].unique()):
        plot_scatter(analysis, model)

    analysis.to_csv(OUT_CSV, index=False)
    print(f"  ✓ {OUT_CSV}")


if __name__ == '__main__':
    main()
