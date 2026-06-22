"""
source_selection_analysis.py
─────────────────────────────
Investigates whether source project properties can guide source selection
for Cross-Project Bug Localization (CPL), and validates a proposed
decision framework against random and oracle baselines.

Rewritten for the 13-project paper set (BLAZE + COOBA results in
obj1_experimental_results.csv). Removes dependency on the old
tranp_cnn_ph1_summary.csv, bug_report_similarity.csv, code_similarity.csv.

Three questions answered
────────────────────────
1. Which features predict good CPL performance (target-side vs source-side)?
2. Can we rank candidate sources for a given target better than random?
3. What is the Hit@K accuracy of a simple heuristic vs oracle?

Output
──────
results/source_selection_validation_BLAZE.csv
results/source_selection_validation_COOBA.csv
results/images/source_selection_*.png

Run
───
    python Scripts/analysis/source_selection_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import kendalltau, spearmanr
from sklearn.preprocessing import MinMaxScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_CSV  = "/home/cs21d002_eashaan/PhD/Objective1/results/paper_results_complete.csv"
METADATA_PKL = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
DOMAIN_CSV   = "/home/cs21d002_eashaan/PhD/Objective1/results/all_project_domain_gaps.csv"
FAISS_DIR    = "/home/cs21d002_eashaan/PhD/Objective1/results/faiss_recall_results"
IMG_DIR      = "/home/cs21d002_eashaan/PhD/Objective1/results/images"
OUT_CSV_TMPL = "/home/cs21d002_eashaan/PhD/Objective1/results/source_selection_validation_{model}.csv"

FAISS_K = 300   # candidate pool size used by TRANP-CNN/COOBA/BLAZE rerankers

os.makedirs(IMG_DIR, exist_ok=True)

MODELS = ['BLAZE', 'COOBA']

# Source features available without bug/code similarity files
SRC_FEATURES = {
    'src_total_unique_bug_reports': 'Source: # bug reports',
    'src_LoC':                      'Source: lines of code',
    'src_code_complexity':          'Source: code complexity',
    'src_bug_report_verbosity':     'Source: bug verbosity',
    'domain_gap':                   'Domain gap (classifier acc.)',
    'loc_ratio':                    'src_LoC / tgt_LoC',
}
TGT_FEATURES = {
    'tgt_LoC':                      'Target: lines of code',
    'tgt_bug_report_verbosity':     'Target: bug verbosity',
    'tgt_total_unique_bug_reports': 'Target: # bug reports',
    'tgt_code_complexity':          'Target: code complexity',
}
ALL_FEATURES = {**TGT_FEATURES, **SRC_FEATURES}


# ── FAISS baseline loader ─────────────────────────────────────────────────────

def load_faiss_recall(k=FAISS_K):
    """
    Returns dict {project_name: Recall@k} for all projects that have a
    FAISS result file. Uses Recall@300 by default (= retrieval ceiling for
    the top-300 candidate pool used by the reranking models).
    """
    col = f'Recall@{k}'
    result = {}
    for fname in os.listdir(FAISS_DIR):
        if not fname.startswith('semantic_search_') or not fname.endswith('_results.csv'):
            continue
        # semantic_search_jupyterlab_jupyterlab_results.csv → jupyterlab/jupyterlab
        stem = fname[len('semantic_search_'):-len('_results.csv')]
        # Only reconstruct known projects (first underscore group = owner)
        try:
            df = pd.read_csv(os.path.join(FAISS_DIR, fname))
            if col in df.columns:
                result[stem.replace('_', '/', 1)] = df[col].iloc[0]
        except Exception:
            pass
    return result


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    df   = pd.read_csv(RESULTS_CSV)
    meta = pd.read_parquet(METADATA_PKL)
    meta_cols = ['repo_name', 'LoC', 'total_unique_bug_reports', 'bug_report_verbosity']
    if 'code_complexity' in meta.columns:
        meta_cols.append('code_complexity')
    meta = meta[meta_cols]

    dg = pd.read_csv(DOMAIN_CSV)

    # Merge source metadata
    src_meta = (meta.rename(columns={'repo_name': 'source_project'})
                    .add_prefix('src_')
                    .rename(columns={'src_source_project': 'source_project'}))
    df = df.merge(src_meta, on='source_project', how='left')

    # Merge target metadata
    tgt_meta = (meta.rename(columns={'repo_name': 'target_project'})
                    .add_prefix('tgt_')
                    .rename(columns={'tgt_target_project': 'target_project'}))
    df = df.merge(tgt_meta, on='target_project', how='left')

    # Domain gap
    dg_map = {}
    for _, r in dg.iterrows():
        dg_map[(r['project_A'], r['project_B'])] = r['domain_gap_accuracy']
        dg_map[(r['project_B'], r['project_A'])] = r['domain_gap_accuracy']
    df['domain_gap'] = df.apply(
        lambda r: dg_map.get((r['source_project'], r['target_project']), np.nan), axis=1)

    df['loc_ratio'] = df['src_LoC'] / (df['tgt_LoC'] + 1e-9)
    return df


def build_cpt(df, model):
    """Build the CP-transfer analysis frame for a single model."""
    g = df[df['model_name'] == model]

    feat_cols = [c for c in ALL_FEATURES if c in g.columns]

    cpt = (g[g['scenario'] == 'CP-transfer']
           [['source_project', 'target_project', 'MRR', 'MAP',
             'top-1', 'top-5', 'top-10'] + feat_cols]
           .copy()
           .rename(columns={'MRR': 'model_mrr', 'MAP': 'model_map'}))

    wps = (g[g['scenario'] == 'WP-small'][['source_project', 'target_project', 'MRR']]
           .rename(columns={'MRR': 'mrr_wps'}))
    wpl = (g[g['scenario'] == 'WP-large'][['source_project', 'target_project', 'MRR']]
           .rename(columns={'MRR': 'mrr_wpl'}))
    cpc = (g[g['scenario'] == 'CP-cold-start'][['source_project', 'target_project', 'MRR']]
           .rename(columns={'MRR': 'mrr_cpc'}))

    cpt = cpt.merge(wps, on=['source_project', 'target_project'], how='left')
    cpt = cpt.merge(wpl, on=['source_project', 'target_project'], how='left')
    cpt = cpt.merge(cpc, on=['source_project', 'target_project'], how='left')

    cpt['cpl_gain'] = cpt['model_mrr'] - cpt['mrr_wps']
    cpt['cpl_wins'] = (cpt['model_mrr'] > cpt['mrr_wps']).astype(int)
    if 'src_total_unique_bug_reports' in cpt.columns:
        cpt['ratio_bugs'] = (cpt['src_total_unique_bug_reports'] /
                             (cpt['tgt_total_unique_bug_reports'] + 1e-9))
    return cpt


# ── Section 1: Feature correlations ──────────────────────────────────────────

def plot_feature_correlations(cpt, model):
    def compute_corrs(target_col):
        rows = []
        for feat, label in ALL_FEATURES.items():
            if feat not in cpt.columns:
                continue
            mask = cpt[feat].notna() & cpt[target_col].notna()
            if mask.sum() < 5:
                continue
            rho, p = spearmanr(cpt.loc[mask, feat], cpt.loc[mask, target_col])
            rows.append({'feature': label, 'rho': rho, 'p': p,
                         'sig': '***' if p < 0.001 else ('**' if p < 0.01 else
                                ('*' if p < 0.05 else ''))})
        return pd.DataFrame(rows).sort_values('rho')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (col, title) in zip(axes, [
        ('model_mrr', f'{model}: Spearman ρ with CP-transfer MRR'),
        ('cpl_gain',  f'{model}: Spearman ρ with CPL gain (CP-transfer − WP-small)'),
    ]):
        corrs = compute_corrs(col)
        if corrs.empty:
            continue
        colors = ['#4CAF50' if r > 0 else '#F44336' for r in corrs['rho']]
        bars = ax.barh(corrs['feature'], corrs['rho'], color=colors, alpha=0.8,
                       edgecolor='black', linewidth=0.5)
        for bar, (_, row) in zip(bars, corrs.iterrows()):
            if row['sig']:
                x = row['rho'] + (0.01 if row['rho'] >= 0 else -0.01)
                ax.text(x, bar.get_y() + bar.get_height() / 2,
                        row['sig'], va='center',
                        ha='left' if row['rho'] >= 0 else 'right',
                        fontsize=10, color='black', fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Spearman ρ', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(-0.9, 0.9)
        ax.grid(axis='x', alpha=0.3)

        tgt_labels = set(TGT_FEATURES.values())
        for bar, (_, row) in zip(bars, corrs.iterrows()):
            if row['feature'] in tgt_labels:
                bar.set_hatch('//')

    handles = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Target property'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Source/pair property'),
        mpatches.Patch(color='#4CAF50', label='Positive ρ'),
        mpatches.Patch(color='#F44336', label='Negative ρ'),
        plt.Line2D([0], [0], color='none', label='*p<0.05 **p<0.01 ***p<0.001'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    out = os.path.join(IMG_DIR, f'source_selection_feature_correlations_{model}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ source_selection_feature_correlations_{model}.png")


# ── Section 2: Target viability profile ───────────────────────────────────────

def plot_target_viability(cpt, model):
    tgt_stats = cpt.groupby('target_project').agg(
        mean_mrr      = ('model_mrr',               'mean'),
        verbosity     = ('tgt_bug_report_verbosity', 'first'),
        loc           = ('tgt_LoC',                 'first'),
        n_bugs        = ('tgt_total_unique_bug_reports', 'first'),
        mean_cpl_gain = ('cpl_gain',                'mean'),
        pct_cpl_wins  = ('cpl_wins',                'mean'),
    ).reset_index()
    tgt_stats['short_name'] = tgt_stats['target_project'].apply(lambda x: x.split('/')[-1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    sc = ax.scatter(tgt_stats['loc'] / 1000, tgt_stats['mean_mrr'],
                    s=tgt_stats['verbosity'] / 2 if tgt_stats['verbosity'].notna().any() else 80,
                    alpha=0.7, c=tgt_stats['mean_mrr'],
                    cmap='RdYlGn', edgecolors='black', linewidth=0.7)
    for _, row in tgt_stats.iterrows():
        ax.annotate(row['short_name'],
                    (row['loc'] / 1000, row['mean_mrr']),
                    textcoords='offset points', xytext=(5, 3), fontsize=8)
    ax.set_xlabel('Target codebase size (kLoC)', fontsize=11)
    ax.set_ylabel('Mean CP-transfer MRR', fontsize=11)
    ax.set_title(f'{model}: Target LoC vs CPL Performance\n(bubble = bug verbosity)',
                 fontsize=11, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Mean CP-transfer MRR')
    ax.grid(alpha=0.3)
    loc_thresh = tgt_stats['loc'].median() / 1000
    ax.axvline(loc_thresh, color='blue', linestyle='--', alpha=0.5,
               label=f'Median LoC ({loc_thresh:.0f}k)')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    sc2 = ax2.scatter(tgt_stats['verbosity'], tgt_stats['mean_mrr'],
                      s=80, alpha=0.8, c=tgt_stats['loc'] / 1000,
                      cmap='YlOrRd_r', edgecolors='black', linewidth=0.7)
    for _, row in tgt_stats.iterrows():
        ax2.annotate(row['short_name'],
                     (row['verbosity'], row['mean_mrr']),
                     textcoords='offset points', xytext=(5, 3), fontsize=8)
    ax2.set_xlabel('Target bug report verbosity (avg words)', fontsize=11)
    ax2.set_ylabel('Mean CP-transfer MRR', fontsize=11)
    ax2.set_title(f'{model}: Bug Verbosity vs CPL Performance\n(colour = codebase size)',
                  fontsize=11, fontweight='bold')
    plt.colorbar(sc2, ax=ax2, label='Target LoC (k)')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(IMG_DIR, f'source_selection_target_viability_{model}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ source_selection_target_viability_{model}.png")
    return tgt_stats


# ── Section 3: Source ranking heuristics ──────────────────────────────────────

def rank_sources(cpt, model):
    """
    For each target, rank available sources by multiple strategies and compare to oracle.

    Strategies
    ──────────
    random        : random ordering (baseline, repeated 1000×)
    heuristic_bugs: rank by src_total_unique_bug_reports (more bugs = better)
    heuristic_comp: rank by src_code_complexity
    heuristic_loc : rank by src_LoC (more code = richer signal)
    composite     : weighted score of src_bugs + src_LoC + domain_gap_inv
    oracle        : actual best source (perfect hindsight)
    """
    score_features = ['src_total_unique_bug_reports', 'src_LoC', 'domain_gap']
    score_features = [f for f in score_features if f in cpt.columns]
    cpt_clean = cpt.dropna(subset=score_features).copy()

    if cpt_clean.empty:
        print(f"  {model}: insufficient data for source ranking — skipping.")
        return pd.DataFrame()

    scaler = MinMaxScaler()
    cpt_clean[score_features] = scaler.fit_transform(cpt_clean[score_features])
    if 'domain_gap' in cpt_clean.columns:
        cpt_clean['domain_gap_inv'] = 1 - cpt_clean['domain_gap']
    else:
        cpt_clean['domain_gap_inv'] = 0.5

    w_bugs = 0.50; w_loc = 0.25; w_gap = 0.25
    cpt_clean['composite'] = (
        cpt_clean.get('src_total_unique_bug_reports', pd.Series(0, index=cpt_clean.index)) * w_bugs +
        cpt_clean.get('src_LoC', pd.Series(0, index=cpt_clean.index))                      * w_loc  +
        cpt_clean['domain_gap_inv']                                                          * w_gap
    )

    results = []
    for tgt, grp in cpt_clean.groupby('target_project'):
        if len(grp) < 3:
            continue
        oracle_mrr = grp['model_mrr'].max()
        oracle_src = grp.loc[grp['model_mrr'].idxmax(), 'source_project']

        rand_mrr = np.mean([grp['model_mrr'].sample(1).values[0] for _ in range(1000)])

        def best_by(col):
            if col not in grp.columns or grp[col].isna().all():
                return np.nan, ''
            idx = grp[col].idxmax()
            return grp.loc[idx, 'model_mrr'], grp.loc[idx, 'source_project']

        bugs_mrr, bugs_src = best_by('src_total_unique_bug_reports')
        comp_mrr, comp_src = best_by('composite')
        loc_mrr,  loc_src  = best_by('src_LoC')

        def tau_score(rank_col):
            if rank_col not in grp.columns or grp[rank_col].isna().all():
                return np.nan, np.nan
            t, p = kendalltau(grp[rank_col].rank(ascending=False), grp['model_mrr'].rank())
            return t, p

        tau_bugs, p_bugs = tau_score('src_total_unique_bug_reports')
        tau_comp, p_comp = tau_score('composite')
        tau_loc,  p_loc  = tau_score('src_LoC')

        results.append({
            'target_project':          tgt,
            'short_name':              tgt.split('/')[-1],
            'n_sources':               len(grp),
            'oracle_mrr':              oracle_mrr,
            'oracle_src':              oracle_src,
            'random_mrr':              rand_mrr,
            'bugs_heuristic_mrr':      bugs_mrr,
            'bugs_heuristic_src':      bugs_src,
            'loc_heuristic_mrr':       loc_mrr,
            'composite_mrr':           comp_mrr,
            'composite_src':           comp_src,
            'pct_oracle_composite':    comp_mrr / (oracle_mrr + 1e-9),
            'tau_bugs':                tau_bugs,
            'tau_composite':           tau_comp,
            'tau_loc':                 tau_loc,
            'hit1_bugs':               int(bugs_src == oracle_src),
            'hit1_composite':          int(comp_src == oracle_src),
        })

    return pd.DataFrame(results)


def plot_source_ranking(val_df, model):
    if val_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x     = np.arange(len(val_df))
    width = 0.18
    ax    = axes[0]
    ax.bar(x - width,   val_df['oracle_mrr'],       width, label='Oracle',          color='gold',    alpha=0.9, edgecolor='black', linewidth=0.6)
    ax.bar(x,           val_df['composite_mrr'],     width, label='Composite',        color='#2196F3', alpha=0.85, edgecolor='black', linewidth=0.6)
    ax.bar(x + width,   val_df['bugs_heuristic_mrr'],width, label='Most bugs',        color='#4CAF50', alpha=0.85, edgecolor='black', linewidth=0.6)
    ax.bar(x + 2*width, val_df['random_mrr'],        width, label='Random baseline', color='#9E9E9E', alpha=0.7,  edgecolor='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(val_df['short_name'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('CP-transfer MRR', fontsize=11)
    ax.set_title(f'{model}: Source Selection Strategy Comparison', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    tau_cols = {
        'Composite':  val_df['tau_composite'].dropna().values,
        'Most bugs':  val_df['tau_bugs'].dropna().values,
        'LoC':        val_df['tau_loc'].dropna().values,
    }
    tau_cols = {k: v for k, v in tau_cols.items() if len(v) > 0}
    if tau_cols:
        bp = ax2.boxplot(tau_cols.values(), labels=tau_cols.keys(),
                         patch_artist=True, notch=False)
        colors_bp = ['#2196F3', '#4CAF50', '#FF9800']
        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Random (τ=0)')
        ax2.set_ylabel('Kendall τ (source ranking quality)', fontsize=11)
        ax2.set_title(f'{model}: Source Ranking Quality\n(τ>0 = better than random)',
                      fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(IMG_DIR, f'source_selection_ranking_quality_{model}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ source_selection_ranking_quality_{model}.png")


# ── Section 4: CPL desirability evidence ─────────────────────────────────────

def print_cpl_desirability_evidence(cpt, model, faiss_recall):
    """
    faiss_recall: dict {project_name: Recall@FAISS_K} from load_faiss_recall().
    Note: FAISS Recall@300 is the retrieval ceiling (fraction of bugs findable
    in top-300 candidates). Model MRR measures ranking quality within those
    candidates. The two metrics differ in scale but together tell whether
    failures are retrieval failures vs ranking failures.
    """
    print(f"\n{'='*60}")
    print(f"CPL DESIRABILITY EVIDENCE — {model}")
    print('='*60)

    total = len(cpt.dropna(subset=['mrr_wps']))

    # E1
    wins = (cpt['model_mrr'] > cpt['mrr_wps']).sum()
    print(f"\nE1  CP-transfer > WP-small: {wins}/{total} pairs ({wins/total:.1%})")

    # E2
    gap = cpt['mrr_wpl'] - cpt['model_mrr']
    gap_valid = gap.dropna()
    n_wpl = len(gap_valid)
    print(f"E2  Mean gap (WP-large − CP-transfer): {gap_valid.mean():.4f} MRR  (n={n_wpl})")
    near = (gap_valid <= 0.02).sum()
    above = (cpt['model_mrr'] >= cpt['mrr_wpl']).sum()
    print(f"    Within 0.02 of WP-large: {near}/{n_wpl} ({near/n_wpl:.1%})")
    print(f"    CP-transfer >= WP-large: {above}/{n_wpl} ({above/n_wpl:.1%})")

    # E3: CP-cold-start MRR vs FAISS Recall@300 per target
    cpt_with_faiss = cpt.copy()
    cpt_with_faiss['faiss_recall'] = cpt_with_faiss['target_project'].map(faiss_recall)
    has_faiss = cpt_with_faiss['faiss_recall'].notna()
    if has_faiss.sum() > 0:
        sub = cpt_with_faiss[has_faiss]
        # Per-target: mean cold-start MRR vs FAISS Recall@300
        tgt_cs = sub.groupby('target_project').agg(
            cs_mrr=('mrr_cpc', 'mean'),
            faiss_r=('faiss_recall', 'first')
        ).dropna()
        n_tgt = len(tgt_cs)
        # Cold-start MRR / FAISS Recall@300 ratio (how much of the ceiling is recovered)
        ratio = tgt_cs['cs_mrr'] / (tgt_cs['faiss_r'] + 1e-9)
        print(f"\nE3  CP-cold-start MRR vs FAISS Recall@{FAISS_K} (retrieval ceiling):")
        print(f"    (Note: MRR and Recall@K differ in scale — ratio shows model/ceiling)")
        print(f"    Mean cold-start MRR across targets:   {tgt_cs['cs_mrr'].mean():.4f}")
        print(f"    Mean FAISS Recall@{FAISS_K} (ceiling): {tgt_cs['faiss_r'].mean():.4f}")
        print(f"    Median cold-start/ceiling ratio:       {ratio.median():.3f}")
        print(f"    Targets where cold-start MRR > 0.10:  "
              f"{(tgt_cs['cs_mrr'] > 0.10).sum()}/{n_tgt}")
    else:
        print(f"\nE3  FAISS results not found for any target in {model} pairs.")

    # E4
    if 'tgt_total_unique_bug_reports' in cpt.columns:
        med = cpt['tgt_total_unique_bug_reports'].median()
        low  = cpt[cpt['tgt_total_unique_bug_reports'] < med]
        high = cpt[cpt['tgt_total_unique_bug_reports'] >= med]
        print(f"\nE4  CPL gain — target FEW bugs  (n={len(low)}):  {low['cpl_gain'].mean():+.4f}")
        print(f"    CPL gain — target MANY bugs (n={len(high)}): {high['cpl_gain'].mean():+.4f}")

    # E5: CP-transfer MRR vs FAISS Recall@300 — shows how much model adds over retrieval
    if has_faiss.sum() > 0:
        sub = cpt_with_faiss[has_faiss].dropna(subset=['model_mrr', 'faiss_recall'])
        ratio_e5 = sub['model_mrr'] / (sub['faiss_recall'] + 1e-9)
        above_ceiling = (sub['model_mrr'] > sub['faiss_recall']).sum()
        print(f"\nE5  CP-transfer MRR vs FAISS Recall@{FAISS_K} — model adds over retrieval:")
        print(f"    Mean CP-transfer MRR:         {sub['model_mrr'].mean():.4f}")
        print(f"    Mean FAISS Recall@{FAISS_K}:   {sub['faiss_recall'].mean():.4f}")
        print(f"    Median model/ceiling ratio:   {ratio_e5.median():.3f}")
        print(f"    Pairs where model MRR > FAISS Recall@{FAISS_K}: "
              f"{above_ceiling}/{len(sub)} ({above_ceiling/len(sub):.1%})")
        print(f"    (model MRR > ceiling indicates model ranks items beyond FAISS top-{FAISS_K})")
    else:
        print(f"\nE5  FAISS results not found for any target in {model} pairs.")

    # E6
    if 'domain_gap' in cpt.columns:
        med_gap = cpt['domain_gap'].median()
        low_g  = cpt[cpt['domain_gap'] < med_gap]
        high_g = cpt[cpt['domain_gap'] >= med_gap]
        if len(low_g) > 0 and len(high_g) > 0:
            print(f"\nE6  CPL wins (low domain gap,  n={len(low_g)}): {low_g['cpl_wins'].mean():.1%}")
            print(f"    CPL wins (high domain gap, n={len(high_g)}): {high_g['cpl_wins'].mean():.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_data()

    faiss_recall = load_faiss_recall(k=FAISS_K)
    n_faiss = len(faiss_recall)
    print(f"  FAISS Recall@{FAISS_K} loaded for {n_faiss} projects: "
          f"{sorted(faiss_recall.keys())[:4]}{'...' if n_faiss > 4 else ''}")

    for model in MODELS:
        model_rows = df[df['model_name'] == model]
        if model_rows.empty:
            print(f"  No data for {model} — skipping.")
            continue

        print(f"\n{'─'*55}")
        print(f"  {model}  ({len(model_rows[model_rows['scenario']=='CP-transfer'])} CP-transfer rows)")
        print(f"{'─'*55}")

        cpt = build_cpt(df, model)
        if cpt.empty:
            print(f"  Could not build CP-transfer frame for {model}.")
            continue
        print(f"  {len(cpt)} CP-transfer rows across {cpt['target_project'].nunique()} targets\n")

        print(f"1. Feature correlations ({model})...")
        plot_feature_correlations(cpt, model)

        print(f"2. Target viability profile ({model})...")
        plot_target_viability(cpt, model)

        print(f"3. Source ranking heuristics ({model})...")
        val_df = rank_sources(cpt, model)
        if not val_df.empty:
            plot_source_ranking(val_df, model)
            out_csv = OUT_CSV_TMPL.format(model=model)
            val_df.to_csv(out_csv, index=False)
            print(f"  ✓ {out_csv} ({len(val_df)} rows)")

            print(f"\n=== Source Ranking Summary — {model} ===")
            pd.set_option('display.float_format', '{:.3f}'.format)
            cols = ['short_name', 'n_sources', 'oracle_mrr', 'composite_mrr',
                    'bugs_heuristic_mrr', 'random_mrr', 'tau_composite', 'tau_bugs', 'hit1_composite']
            cols = [c for c in cols if c in val_df.columns]
            print(val_df[cols].to_string(index=False))
            print(f"\nOverall Hit@1 (composite):  {val_df['hit1_composite'].mean():.1%}")
            print(f"Overall Hit@1 (most bugs):  {val_df['hit1_bugs'].mean():.1%}")
            print(f"Mean Kendall τ (composite): {val_df['tau_composite'].mean():.3f}")
            print(f"Mean Kendall τ (most bugs): {val_df['tau_bugs'].mean():.3f}")
            valid_tau = val_df['tau_composite'].dropna()
            print(f"Targets with τ > 0 (composite): {(valid_tau > 0).sum()}/{len(valid_tau)}")

        print_cpl_desirability_evidence(cpt, model, faiss_recall)

    print(f"\n✓ Done.")


if __name__ == '__main__':
    main()
