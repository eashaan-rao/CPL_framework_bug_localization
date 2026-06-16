"""
negative_transfer_analysis.py
─────────────────────────────
Characterizes when CPL hurts (negative transfer) vs helps, across BLAZE and COOBA.

Questions answered
──────────────────
1. What observable features predict the sign and magnitude of CP-transfer − WP-small?
2. Which pairs exhibit negative transfer, and what do they share?
3. For symmetric pairs (A→B and B→A), does the smaller-LoC target consistently benefit?

Outputs
───────
results/negative_transfer_analysis.csv              — per-pair deltas + features + transfer label
results/negative_transfer_analysis_commutativity.csv — symmetric pair breakdown
results/images/nt_feature_correlations.png
results/images/nt_delta_by_target_loc.png
results/images/nt_commutativity.png

Run
───
    python Scripts/analysis/negative_transfer_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, wilcoxon

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_CSV  = "/home/cs21d002_eashaan/PhD/Objective1/results/obj1_paper_results.csv"
METADATA_PKL = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
DOMAIN_CSV   = "/home/cs21d002_eashaan/PhD/Objective1/results/all_project_domain_gaps.csv"
IMG_DIR      = "/home/cs21d002_eashaan/PhD/Objective1/results/images"
OUT_CSV           = "/home/cs21d002_eashaan/PhD/Objective1/results/negative_transfer_analysis.csv"
OUT_COMM_CSV      = "/home/cs21d002_eashaan/PhD/Objective1/results/negative_transfer_analysis_commutativity.csv"
OUT_CROSSDOMAIN_CSV = "/home/cs21d002_eashaan/PhD/Objective1/results/cross_domain_breakdown.csv"

# The 6 DS projects form the within-domain group (Group A pairs in paper set).
DS_PROJECTS = {
    'jupyterlab/jupyterlab',
    'lightning-ai/lightning',
    'prefecthq/prefect',
    'pydata/xarray',
    'numpy/numpy',
    'scikit-learn/scikit-learn',
}

os.makedirs(IMG_DIR, exist_ok=True)

FEATURES = {
    'tgt_LoC':                      'Target: LoC',
    'src_LoC':                      'Source: LoC',
    'tgt_total_unique_bug_reports': 'Target: # bug reports',
    'src_total_unique_bug_reports': 'Source: # bug reports',
    'loc_ratio':                    'src_LoC / tgt_LoC',
    'tgt_bug_report_verbosity':     'Target: bug verbosity',
    'src_bug_report_verbosity':     'Source: bug verbosity',
    'domain_gap':                   'Domain gap',
}

POS_THRESH =  0.01
NEG_THRESH = -0.01


def load_data():
    df   = pd.read_csv(RESULTS_CSV)
    meta = pd.read_parquet(METADATA_PKL)[['repo_name', 'LoC', 'total_unique_bug_reports',
                                          'bug_report_verbosity', 'code_complexity']]
    dg   = pd.read_csv(DOMAIN_CSV)

    df = df.merge(
        meta.rename(columns={'repo_name': 'source_project'}).add_prefix('src_')
            .rename(columns={'src_source_project': 'source_project'}),
        on='source_project', how='left')
    df = df.merge(
        meta.rename(columns={'repo_name': 'target_project'}).add_prefix('tgt_')
            .rename(columns={'tgt_target_project': 'target_project'}),
        on='target_project', how='left')

    dg_map = {}
    for _, r in dg.iterrows():
        dg_map[(r['project_A'], r['project_B'])] = r['domain_gap_accuracy']
        dg_map[(r['project_B'], r['project_A'])] = r['domain_gap_accuracy']
    df['domain_gap'] = df.apply(
        lambda r: dg_map.get((r['source_project'], r['target_project']), np.nan), axis=1)

    df['loc_ratio'] = df['src_LoC'] / (df['tgt_LoC'] + 1e-9)
    return df


def build_delta_df(df):
    feat_cols = [c for c in FEATURES if c in df.columns]
    cpt = df[df['scenario'] == 'CP-transfer'][
        ['model_name', 'source_project', 'target_project', 'MRR', 'MAP'] + feat_cols
    ].copy()
    wps = df[df['scenario'] == 'WP-small'][
        ['model_name', 'source_project', 'target_project', 'MRR', 'MAP']
    ].rename(columns={'MRR': 'mrr_wps', 'MAP': 'map_wps'})

    delta = cpt.merge(wps, on=['model_name', 'source_project', 'target_project'])
    delta['delta_mrr'] = delta['MRR'] - delta['mrr_wps']
    delta['delta_map'] = delta['MAP'] - delta['map_wps']
    delta['transfer']  = delta['delta_mrr'].apply(
        lambda d: 'positive' if d > POS_THRESH else ('negative' if d < NEG_THRESH else 'neutral'))
    return delta


def plot_feature_correlations(delta):
    models = sorted(delta['model_name'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(9 * len(models), 7))
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    for ax, model in zip(axes, models):
        sub  = delta[delta['model_name'] == model]
        rhos, colors, labels_out = [], [], []
        for feat, label in FEATURES.items():
            if feat not in sub.columns or sub[feat].isna().all():
                continue
            mask = sub[feat].notna() & sub['delta_mrr'].notna()
            if mask.sum() < 5:
                continue
            rho, p = spearmanr(sub.loc[mask, feat], sub.loc[mask, 'delta_mrr'])
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            rhos.append(rho)
            colors.append('#4CAF50' if rho > 0 else '#F44336')
            labels_out.append(f"{label}  {sig}".strip())

        order = np.argsort(rhos)
        ax.barh(np.array(labels_out)[order], np.array(rhos)[order],
                color=np.array(colors)[order], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Spearman ρ with CPL gain (delta MRR)', fontsize=10)
        ax.set_title(f'{model}: Features predicting CPL gain', fontsize=11, fontweight='bold')
        ax.set_xlim(-1.0, 1.0)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'nt_feature_correlations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ nt_feature_correlations.png")


def plot_delta_by_target_loc(delta):
    models = sorted(delta['model_name'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5))
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    color_map = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}

    for ax, model in zip(axes, models):
        sub = delta[delta['model_name'] == model].dropna(subset=['tgt_LoC', 'delta_mrr'])
        colors = sub['transfer'].map(color_map)
        ax.scatter(sub['tgt_LoC'] / 1000, sub['delta_mrr'],
                   c=colors, alpha=0.78, edgecolors='k', linewidth=0.4, s=65)
        ax.axhline(0,          color='black',   linewidth=0.9, linestyle='--')
        ax.axhline(POS_THRESH, color='#4CAF50', linewidth=0.6, linestyle=':')
        ax.axhline(NEG_THRESH, color='#F44336', linewidth=0.6, linestyle=':')
        ax.set_xlabel('Target LoC (thousands)', fontsize=10)
        ax.set_ylabel('CPL gain: CP-transfer − WP-small (MRR)', fontsize=10)
        ax.set_title(f'{model}: CPL gain vs target codebase size', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)

        n_neg = (sub['transfer'] == 'negative').sum()
        ax.text(0.98, 0.98, f'Negative transfer: {n_neg}/{len(sub)}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    handles = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in color_map.items()]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'nt_delta_by_target_loc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ nt_delta_by_target_loc.png")


def commutativity_analysis(delta):
    rows = []
    for model, g in delta.groupby('model_name'):
        pair_set = set(zip(g['source_project'], g['target_project']))
        seen = set()
        for src, tgt in pair_set:
            if (tgt, src) in pair_set and (tgt, src) not in seen:
                fwd = g[(g['source_project'] == src) & (g['target_project'] == tgt)].iloc[0]
                rev = g[(g['source_project'] == tgt) & (g['target_project'] == src)].iloc[0]
                smaller_tgt_benefits = (
                    (fwd['tgt_LoC'] < rev['tgt_LoC']) == (fwd['delta_mrr'] > rev['delta_mrr'])
                    if not (pd.isna(fwd['tgt_LoC']) or pd.isna(rev['tgt_LoC']))
                    else None
                )
                rows.append({
                    'model': model,
                    'project_A': src, 'project_B': tgt,
                    'delta_A_to_B': fwd['delta_mrr'],
                    'delta_B_to_A': rev['delta_mrr'],
                    'tgt_LoC_A_to_B': fwd.get('tgt_LoC', np.nan),
                    'tgt_LoC_B_to_A': rev.get('tgt_LoC', np.nan),
                    'smaller_tgt_benefits': smaller_tgt_benefits,
                })
                seen.add((src, tgt))

    comm_df = pd.DataFrame(rows)
    if comm_df.empty:
        print("  No symmetric pairs found.")
        return comm_df

    models = comm_df['model'].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 6))
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    for ax, (model, g) in zip(axes, comm_df.groupby('model')):
        g = g.dropna(subset=['delta_A_to_B', 'delta_B_to_A'])
        color_fn = lambda v: '#4CAF50' if v is True else ('#F44336' if v is False else '#9E9E9E')
        colors = g['smaller_tgt_benefits'].apply(color_fn)
        ax.scatter(g['delta_A_to_B'], g['delta_B_to_A'],
                   c=colors, s=80, edgecolors='k', linewidth=0.4, alpha=0.85)
        for _, row in g.iterrows():
            ax.annotate(row['project_B'].split('/')[-1],
                        (row['delta_A_to_B'], row['delta_B_to_A']),
                        fontsize=7, ha='left', va='bottom', alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
        ax.set_xlabel('CPL gain: A → B (delta MRR)', fontsize=10)
        ax.set_ylabel('CPL gain: B → A (delta MRR)', fontsize=10)
        ax.set_title(f'{model}: CPL commutativity', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        n_consistent = g['smaller_tgt_benefits'].sum()
        ax.text(0.02, 0.97,
                f'Smaller-target benefits more: {int(n_consistent)}/{len(g)}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    handles = [
        mpatches.Patch(color='#4CAF50', label='Consistent: smaller target benefits more'),
        mpatches.Patch(color='#F44336', label='Inconsistent direction'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'nt_commutativity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ nt_commutativity.png")
    return comm_df


def cross_domain_breakdown(delta):
    """
    Compare CPL gain for within-domain (DS×DS) vs cross-domain pairs.
    Tests E6 from CPL_DESIRABILITY: domain gap does not prevent CPL benefit.
    """
    delta = delta.copy()
    delta['domain_type'] = delta.apply(
        lambda r: 'Within-domain (DS×DS)'
        if r['source_project'] in DS_PROJECTS and r['target_project'] in DS_PROJECTS
        else 'Cross-domain',
        axis=1
    )

    rows = []
    for model, mg in delta.groupby('model_name'):
        for dtype, dg in mg.groupby('domain_type'):
            n = len(dg)
            n_pos = (dg['transfer'] == 'positive').sum()
            n_neg = (dg['transfer'] == 'negative').sum()
            rows.append({
                'model': model,
                'domain_type': dtype,
                'n_pairs': n,
                'mean_delta_mrr': dg['delta_mrr'].mean(),
                'median_delta_mrr': dg['delta_mrr'].median(),
                'win_rate': n_pos / n if n > 0 else np.nan,
                'n_positive': int(n_pos),
                'n_negative': int(n_neg),
            })
    cd_df = pd.DataFrame(rows)

    print("\n── Cross-domain vs Within-domain CPL Gain ───────────────────────────")
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(cd_df[['model', 'domain_type', 'n_pairs', 'mean_delta_mrr',
                 'win_rate', 'n_positive', 'n_negative']].to_string(index=False))

    # Plot: side-by-side boxplot per model
    models = sorted(delta['model_name'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5))
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    palette = {'Within-domain (DS×DS)': '#2196F3', 'Cross-domain': '#FF9800'}
    for ax, model in zip(axes, models):
        sub = delta[delta['model_name'] == model].dropna(subset=['delta_mrr'])
        groups = ['Within-domain (DS×DS)', 'Cross-domain']
        data   = [sub[sub['domain_type'] == g]['delta_mrr'].values for g in groups]
        bp = ax.boxplot(data, labels=[g.replace(' (DS×DS)', '\n(DS×DS)') for g in groups],
                        patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], [palette[g] for g in groups]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(0, color='red', linewidth=1.2, linestyle='--', alpha=0.8, label='Zero gain')
        ax.set_ylabel('CPL gain: CP-transfer − WP-small (MRR)', fontsize=10)
        ax.set_title(f'{model}: CPL gain by domain type', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, (g, d) in enumerate(zip(groups, data), start=1):
            n_win  = (d > 0).sum()
            n_loss = (d < 0).sum()
            ax.text(i, ax.get_ylim()[0] + 0.01,
                    f'n={len(d)}\nwin={n_win} loss={n_loss}',
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'nt_cross_domain_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ nt_cross_domain_breakdown.png")
    return cd_df


def print_summary(delta):
    print("\n── Negative Transfer Summary ─────────────────────────────────────────")
    for model, g in delta.groupby('model_name'):
        n_pos = (g['transfer'] == 'positive').sum()
        n_neg = (g['transfer'] == 'negative').sum()
        n_neu = (g['transfer'] == 'neutral').sum()
        print(f"  {model}: n={len(g)}  positive={n_pos}  neutral={n_neu}  negative={n_neg}")
        print(f"    delta MRR: mean={g['delta_mrr'].mean():.3f}  "
              f"median={g['delta_mrr'].median():.3f}  "
              f"min={g['delta_mrr'].min():.3f}  max={g['delta_mrr'].max():.3f}")
        d = g['delta_mrr'].dropna()
        if len(d) > 1:
            _, p = wilcoxon(d, alternative='greater', zero_method='zsplit')
            print(f"    Wilcoxon (delta > 0): p={p:.4f}")
        if n_neg > 0:
            neg_cases = g[g['transfer'] == 'negative'][
                ['source_project', 'target_project', 'delta_mrr', 'tgt_LoC']
            ].sort_values('delta_mrr')
            print(f"    Negative transfer cases:")
            for _, row in neg_cases.iterrows():
                loc_str = f"  tgt_LoC={row['tgt_LoC']/1000:.0f}K" if not pd.isna(row.get('tgt_LoC')) else ''
                print(f"      {row['source_project']} → {row['target_project']}: "
                      f"delta={row['delta_mrr']:.3f}{loc_str}")
        print()


def main():
    print("Loading data...")
    df    = load_data()
    delta = build_delta_df(df)

    print_summary(delta)
    plot_feature_correlations(delta)
    plot_delta_by_target_loc(delta)
    comm_df = commutativity_analysis(delta)
    cd_df   = cross_domain_breakdown(delta)

    delta.to_csv(OUT_CSV, index=False)
    print(f"  ✓ {OUT_CSV}")
    if not comm_df.empty:
        comm_df.to_csv(OUT_COMM_CSV, index=False)
        print(f"  ✓ {OUT_COMM_CSV}")
    if not cd_df.empty:
        cd_df.to_csv(OUT_CROSSDOMAIN_CSV, index=False)
        print(f"  ✓ {OUT_CROSSDOMAIN_CSV}")


if __name__ == '__main__':
    main()
