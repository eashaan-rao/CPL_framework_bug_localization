"""
cross_model_consistency_analysis.py
─────────────────────────────────────
Tests whether BLAZE and COOBA agree on which pairs benefit from CPL.

Questions answered
──────────────────
1. For pairs tested by both models, do they agree on the direction of CPL gain?
2. How large is the disagreement in magnitude when it occurs?
3. Is agreement higher for certain target conditions (small vs large)?

Outputs
───────
results/cross_model_agreement.csv           — per-pair deltas, direction labels, agreement flag
results/images/cm_agreement_scatter.png     — BLAZE delta vs COOBA delta scatter
results/images/cm_delta_comparison.png      — side-by-side bar chart sorted by target LoC

Run
───
    python Scripts/analysis/cross_model_consistency_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_CSV  = "/home/cs21d002_eashaan/PhD/Objective1/results/paper_results_complete_corrected.csv"
METADATA_PKL = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
IMG_DIR      = "/home/cs21d002_eashaan/PhD/Objective1/results/images"
OUT_CSV      = "/home/cs21d002_eashaan/PhD/Objective1/results/cross_model_agreement.csv"

os.makedirs(IMG_DIR, exist_ok=True)

THRESHOLD = 0.01  # neutral band: |delta| < THRESHOLD treated as no clear direction


def load_data():
    df   = pd.read_csv(RESULTS_CSV)
    meta = pd.read_parquet(METADATA_PKL)[['repo_name', 'LoC', 'total_unique_bug_reports']]
    df   = df.merge(
        meta.rename(columns={'repo_name': 'target_project', 'LoC': 'tgt_LoC',
                              'total_unique_bug_reports': 'tgt_n_bugs'}),
        on='target_project', how='left')
    return df


def build_agreement_df(df):
    per_model = {}
    for model in ['BLAZE', 'COOBA']:
        g   = df[df['model_name'] == model]
        cpt = g[g['scenario'] == 'CP-transfer'].set_index(['source_project', 'target_project'])['MRR']
        wps = g[g['scenario'] == 'WP-small'].set_index(['source_project', 'target_project'])['MRR']
        common = cpt.index.intersection(wps.index)
        per_model[model] = (cpt[common] - wps[common]).rename(f'delta_{model}')

    blaze_d = per_model['BLAZE']
    cooba_d = per_model['COOBA']
    common  = blaze_d.index.intersection(cooba_d.index)

    merged = pd.DataFrame({
        'delta_BLAZE': blaze_d[common],
        'delta_COOBA': cooba_d[common],
    }).reset_index()

    def direction(d):
        return 'positive' if d > THRESHOLD else ('negative' if d < -THRESHOLD else 'neutral')

    merged['dir_BLAZE']  = merged['delta_BLAZE'].apply(direction)
    merged['dir_COOBA']  = merged['delta_COOBA'].apply(direction)
    merged['agree']      = merged['dir_BLAZE'] == merged['dir_COOBA']
    merged['agree_sign'] = np.sign(merged['delta_BLAZE']) == np.sign(merged['delta_COOBA'])

    meta = pd.read_parquet(METADATA_PKL)[['repo_name', 'LoC', 'total_unique_bug_reports']]
    merged = merged.merge(
        meta.rename(columns={'repo_name': 'target_project', 'LoC': 'tgt_LoC',
                              'total_unique_bug_reports': 'tgt_n_bugs'}),
        on='target_project', how='left')
    return merged


def plot_scatter(merged):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors  = merged['agree'].map({True: '#4CAF50', False: '#F44336'})
    ax.scatter(merged['delta_BLAZE'], merged['delta_COOBA'],
               c=colors, s=80, edgecolors='k', linewidth=0.5, alpha=0.85, zorder=3)

    for _, row in merged.iterrows():
        ax.annotate(row['target_project'].split('/')[-1],
                    (row['delta_BLAZE'], row['delta_COOBA']),
                    fontsize=7, ha='left', va='bottom', alpha=0.75)

    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline( THRESHOLD, color='grey', linewidth=0.5, linestyle=':', alpha=0.6)
    ax.axhline(-THRESHOLD, color='grey', linewidth=0.5, linestyle=':', alpha=0.6)
    ax.axvline( THRESHOLD, color='grey', linewidth=0.5, linestyle=':', alpha=0.6)
    ax.axvline(-THRESHOLD, color='grey', linewidth=0.5, linestyle=':', alpha=0.6)

    ax.set_xlabel('BLAZE CPL gain (CP-transfer − WP-small, MRR)', fontsize=11)
    ax.set_ylabel('COOBA CPL gain (CP-transfer − WP-small, MRR)', fontsize=11)
    ax.set_title('Cross-model CPL gain agreement\n(shared source→target pairs)',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    rho, p = spearmanr(merged['delta_BLAZE'], merged['delta_COOBA'])
    n_agree = merged['agree'].sum()
    n = len(merged)
    ax.text(0.02, 0.98,
            f"Direction agreement: {n_agree}/{n} ({100*n_agree/n:.0f}%)\n"
            f"Spearman ρ = {rho:.2f}  (p={p:.3f})",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    handles = [
        mpatches.Patch(color='#4CAF50', label='Models agree on CPL direction'),
        mpatches.Patch(color='#F44336', label='Models disagree'),
    ]
    ax.legend(handles=handles, fontsize=9, loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'cm_agreement_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ cm_agreement_scatter.png")


def plot_delta_comparison(merged):
    sub = merged.dropna(subset=['tgt_LoC']).sort_values('tgt_LoC').reset_index(drop=True)
    x   = np.arange(len(sub))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(sub) * 0.6), 5))
    ax.bar(x - w/2, sub['delta_BLAZE'], w,
           label='BLAZE', color='#2196F3', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, sub['delta_COOBA'], w,
           label='COOBA', color='#FF9800', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['target_project'].split('/')[-1]}\n({int(r['tgt_LoC']/1000)}K LoC)"
         for _, r in sub.iterrows()],
        rotation=45, ha='right', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('CPL gain: CP-transfer − WP-small (MRR)', fontsize=10)
    ax.set_title('Per-target CPL gain by model architecture\n(sorted by target LoC)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'cm_delta_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ cm_delta_comparison.png")


def build_pairwise_table(df):
    """Direction agreement + Spearman rho for all three model-pair combinations
    (feeds tab:model_agreement in the paper)."""
    delta = {}
    for model in ['BLAZE', 'COOBA', 'TRANP-CNN']:
        g   = df[df['model_name'] == model]
        cpt = g[g['scenario'] == 'CP-transfer'].set_index(['source_project', 'target_project'])['MRR']
        wps = g[g['scenario'] == 'WP-small'].set_index(['source_project', 'target_project'])['MRR']
        common = cpt.index.intersection(wps.index)
        delta[model] = (cpt[common] - wps[common])

    def direction(d):
        return 'positive' if d > THRESHOLD else ('negative' if d < -THRESHOLD else 'neutral')

    rows = []
    for m1, m2 in [('BLAZE', 'COOBA'), ('BLAZE', 'TRANP-CNN'), ('COOBA', 'TRANP-CNN')]:
        idx = delta[m1].index.intersection(delta[m2].index)
        x, y = delta[m1][idx], delta[m2][idx]
        agree = (x.apply(direction) == y.apply(direction)).mean()
        rho, p = spearmanr(x, y)
        rows.append({'pair': f'{m1} vs. {m2}', 'n': len(idx),
                      'agreement_pct': 100 * agree, 'spearman_rho': rho, 'p_value': p})

    table = pd.DataFrame(rows)
    print("\n── Pairwise Cross-Model Direction Agreement (all 3 combinations) ──────")
    for _, r in table.iterrows():
        sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01 else
              ('*' if r['p_value'] < 0.05 else 'n.s.'))
        print(f"  {r['pair']:22s} n={r['n']:.0f}  agreement={r['agreement_pct']:.0f}%  "
              f"rho={r['spearman_rho']:.3f} {sig} (p={r['p_value']:.4f})")
    out = "/home/cs21d002_eashaan/PhD/Objective1/results/cross_model_agreement_pairwise.csv"
    table.to_csv(out, index=False)
    print(f"  ✓ {out}")
    return table


def print_summary(merged):
    n = len(merged)
    n_agree      = merged['agree'].sum()
    n_agree_sign = merged['agree_sign'].sum()
    print(f"\n── Cross-Model Agreement Summary ─────────────────────────────────────")
    print(f"  Shared pairs: {n}")
    print(f"  Direction agreement (neutral band ±{THRESHOLD}): {n_agree}/{n} ({100*n_agree/n:.0f}%)")
    print(f"  Sign agreement (strict, no neutral band):         {n_agree_sign}/{n} ({100*n_agree_sign/n:.0f}%)")
    rho, p = spearmanr(merged['delta_BLAZE'], merged['delta_COOBA'])
    print(f"  Spearman ρ between model deltas: {rho:.3f}  (p={p:.4f})")

    disagree = merged[~merged['agree']].sort_values('delta_COOBA')
    if len(disagree) > 0:
        print(f"\n  Disagreement cases ({len(disagree)}):")
        for _, row in disagree.iterrows():
            print(f"    {row['source_project']} → {row['target_project']}: "
                  f"BLAZE={row['delta_BLAZE']:+.3f}  COOBA={row['delta_COOBA']:+.3f}")
    print()


def main():
    print("Loading data...")
    df     = load_data()
    merged = build_agreement_df(df)

    print_summary(merged)
    plot_scatter(merged)
    plot_delta_comparison(merged)
    build_pairwise_table(df)

    merged.to_csv(OUT_CSV, index=False)
    print(f"  ✓ {OUT_CSV}")


if __name__ == '__main__':
    main()
