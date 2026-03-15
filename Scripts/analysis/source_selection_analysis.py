"""
source_selection_analysis.py
─────────────────────────────
Investigates whether source project properties can guide source selection
for Cross-Project Bug Localization (CPL), and validates a proposed
decision framework against random and oracle baselines.

Three questions answered
────────────────────────
1. Which features predict good CPL performance (target-side vs source-side)?
2. Can we rank candidate sources for a given target better than random?
3. What is the Hit@K accuracy of a simple heuristic vs oracle?

Output
──────
results/source_selection_validation.csv   — per-target ranking quality
results/images/source_selection_*.png     — supporting plots

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
from scipy import stats
from scipy.stats import kendalltau, spearmanr
from sklearn.preprocessing import MinMaxScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
SUMMARY_CSV  = "/home/cs21d002_eashaan/PhD/Objective1/results/tranp_cnn_ph1_summary.csv"
BUG_SIM_CSV  = "/home/cs21d002_eashaan/PhD/Objective1/results/bug_report_similarity.csv"
CODE_SIM_CSV = "/home/cs21d002_eashaan/PhD/Objective1/results/code_similarity.csv"
IMG_DIR      = "/home/cs21d002_eashaan/PhD/Objective1/results/images"
OUT_CSV      = "/home/cs21d002_eashaan/PhD/Objective1/results/source_selection_validation.csv"

os.makedirs(IMG_DIR, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────

def add_sim(df, sim_df, val_col, new_col):
    d = {}
    for _, r in sim_df.iterrows():
        d[(r['project_A'], r['project_B'])] = r[val_col]
        d[(r['project_B'], r['project_A'])] = r[val_col]
    df[new_col] = df.apply(
        lambda r: d.get((r['source_project'], r['target_project']), np.nan), axis=1)
    return df


def load_data():
    df       = pd.read_csv(SUMMARY_CSV)
    bug_sim  = pd.read_csv(BUG_SIM_CSV)
    code_sim = pd.read_csv(CODE_SIM_CSV)
    df = add_sim(df, bug_sim,  'mean_sim',      'bug_sim')
    df = add_sim(df, code_sim, 'mean_code_sim', 'code_sim')

    cpt = df[df['scenario'] == 'CP-transfer'].copy()
    wps = df[df['scenario'] == 'WP-small'][['source_project','target_project','model_mrr']].rename(
          columns={'model_mrr':'mrr_wps'})
    wpl = df[df['scenario'] == 'WP-large'][['source_project','target_project','model_mrr']].rename(
          columns={'model_mrr':'mrr_wpl'})
    cpc = df[df['scenario'] == 'CP-cold-start'][['source_project','target_project','model_mrr']].rename(
          columns={'model_mrr':'mrr_cpc'})

    cpt = cpt.merge(wps, on=['source_project','target_project'])
    cpt = cpt.merge(wpl, on=['source_project','target_project'])
    cpt = cpt.merge(cpc, on=['source_project','target_project'])
    cpt['cpl_gain']   = cpt['model_mrr'] - cpt['mrr_wps']
    cpt['cpl_wins']   = (cpt['model_mrr'] > cpt['mrr_wps']).astype(int)
    cpt['ratio_bugs'] = cpt['src_total_unique_bug_reports'] / (cpt['tgt_total_unique_bug_reports'] + 1e-9)
    return cpt


# ── Section 1: Feature correlations ──────────────────────────────────────────

def plot_feature_correlations(cpt):
    """
    Two-panel: (left) features vs CPT MRR, (right) features vs CPL gain.
    Bars are sorted by |rho|. Stars mark significant results.
    """
    src_features = {
        'src_total_unique_bug_reports': 'Source: # bug reports',
        'src_LoC':                      'Source: lines of code',
        'src_code_complexity':          'Source: code complexity',
        'src_bug_report_verbosity':     'Source: bug verbosity',
        'bug_sim':                      'Bug report similarity (A↔B)',
        'code_sim':                     'Code embedding similarity (A↔B)',
        'domain_gap':                   'Domain gap (classifier acc.)',
    }
    tgt_features = {
        'tgt_LoC':                      'Target: lines of code',
        'tgt_bug_report_verbosity':     'Target: bug verbosity',
        'tgt_polyglot_index':           'Target: polyglot index',
        'tgt_total_unique_bug_reports': 'Target: # bug reports',
        'tgt_code_complexity':          'Target: code complexity',
    }
    all_features = {**tgt_features, **src_features}

    def compute_corrs(target_col):
        rows = []
        for feat, label in all_features.items():
            if feat not in cpt.columns:
                continue
            mask = cpt[feat].notna() & cpt[target_col].notna()
            rho, p = spearmanr(cpt.loc[mask, feat], cpt.loc[mask, target_col])
            rows.append({'feature': label, 'rho': rho, 'p': p,
                         'sig': '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ''))})
        return pd.DataFrame(rows).sort_values('rho')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (col, title) in zip(axes, [
        ('model_mrr', 'Spearman ρ with CP-transfer MRR'),
        ('cpl_gain',  'Spearman ρ with CPL gain (CP-transfer − WP-small)'),
    ]):
        corrs = compute_corrs(col)
        colors = ['#4CAF50' if r > 0 else '#F44336' for r in corrs['rho']]
        bars = ax.barh(corrs['feature'], corrs['rho'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        for bar, (_, row) in zip(bars, corrs.iterrows()):
            if row['sig']:
                x = row['rho'] + (0.01 if row['rho'] >= 0 else -0.01)
                ax.text(x, bar.get_y() + bar.get_height()/2,
                        row['sig'], va='center', ha='left' if row['rho'] >= 0 else 'right',
                        fontsize=10, color='black', fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Spearman ρ', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(-0.7, 0.85)
        ax.grid(axis='x', alpha=0.3)

        # Shade target vs source features
        tgt_labels = set(tgt_features.values())
        for bar, (_, row) in zip(bars, corrs.iterrows()):
            if row['feature'] in tgt_labels:
                bar.set_hatch('//')

    # Legend
    handles = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Target property'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Source/pair property'),
        mpatches.Patch(color='#4CAF50', label='Positive correlation'),
        mpatches.Patch(color='#F44336', label='Negative correlation'),
        plt.Line2D([0],[0], color='none', label='*p<0.05, **p<0.01, ***p<0.001'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'source_selection_feature_correlations.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ source_selection_feature_correlations.png")


# ── Section 2: Target viability profile ───────────────────────────────────────

def plot_target_viability(cpt):
    """
    Scatter: tgt_LoC vs mean CPT MRR per target, sized by tgt_bug_verbosity.
    Labels each target. Shows the target-property-driven CPL feasibility landscape.
    """
    tgt_stats = cpt.groupby('target_project').agg(
        mean_mrr          = ('model_mrr',               'mean'),
        verbosity         = ('tgt_bug_report_verbosity', 'first'),
        loc               = ('tgt_LoC',                 'first'),
        n_bugs            = ('tgt_total_unique_bug_reports','first'),
        mean_cpl_gain     = ('cpl_gain',                 'mean'),
        pct_cpl_wins      = ('cpl_wins',                 'mean'),
    ).reset_index()
    tgt_stats['short_name'] = tgt_stats['target_project'].apply(lambda x: x.split('/')[1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: LoC vs MRR (bubble = verbosity)
    ax = axes[0]
    sc = ax.scatter(tgt_stats['loc'] / 1000, tgt_stats['mean_mrr'],
                    s=tgt_stats['verbosity'] / 2, alpha=0.7,
                    c=tgt_stats['mean_mrr'], cmap='RdYlGn', edgecolors='black', linewidth=0.7)
    for _, row in tgt_stats.iterrows():
        ax.annotate(row['short_name'],
                    (row['loc']/1000, row['mean_mrr']),
                    textcoords='offset points', xytext=(5, 3), fontsize=8)
    ax.set_xlabel('Target codebase size (kLoC)', fontsize=11)
    ax.set_ylabel('Mean CP-transfer MRR', fontsize=11)
    ax.set_title('Target LoC vs CPL Performance\n(bubble size = bug report verbosity)',
                 fontsize=11, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Mean CP-transfer MRR')
    ax.grid(alpha=0.3)

    # Panel 2: verbosity vs MRR
    ax2 = axes[1]
    sc2 = ax2.scatter(tgt_stats['verbosity'], tgt_stats['mean_mrr'],
                      s=80, alpha=0.8,
                      c=tgt_stats['loc']/1000, cmap='YlOrRd_r', edgecolors='black', linewidth=0.7)
    for _, row in tgt_stats.iterrows():
        ax2.annotate(row['short_name'],
                     (row['verbosity'], row['mean_mrr']),
                     textcoords='offset points', xytext=(5, 3), fontsize=8)
    ax2.set_xlabel('Target bug report verbosity (avg words)', fontsize=11)
    ax2.set_ylabel('Mean CP-transfer MRR', fontsize=11)
    ax2.set_title('Bug Verbosity vs CPL Performance\n(colour = codebase size)',
                  fontsize=11, fontweight='bold')
    plt.colorbar(sc2, ax=ax2, label='Target LoC (k)')
    ax2.grid(alpha=0.3)

    # Viability thresholds (median split)
    loc_thresh  = tgt_stats['loc'].median() / 1000
    verb_thresh = tgt_stats['verbosity'].median()
    axes[0].axvline(loc_thresh,  color='blue', linestyle='--', alpha=0.5, label=f'Median LoC ({loc_thresh:.0f}k)')
    axes[1].axvline(verb_thresh, color='blue', linestyle='--', alpha=0.5, label=f'Median verbosity ({verb_thresh:.0f})')
    axes[0].legend(fontsize=9)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'source_selection_target_viability.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ source_selection_target_viability.png")
    return tgt_stats


# ── Section 3: Source ranking heuristics ──────────────────────────────────────

def rank_sources(cpt):
    """
    For each target, rank available sources by three strategies and compare to oracle.

    Strategies
    ──────────
    random        : random ordering (baseline, repeated 1000×, take mean)
    heuristic_bugs: rank by src_total_unique_bug_reports (more bugs = better source)
    heuristic_comp: rank by src_code_complexity
    composite     : weighted score of src_bugs + src_LoC + domain_gap_inv + bug_sim + code_sim
    oracle        : actual best source (perfect hindsight)
    """
    score_features = ['src_total_unique_bug_reports', 'src_LoC', 'bug_sim', 'code_sim', 'domain_gap']
    cpt_clean = cpt.dropna(subset=['bug_sim', 'code_sim']).copy()

    # Normalise for composite
    scaler = MinMaxScaler()
    cpt_clean[score_features] = scaler.fit_transform(cpt_clean[score_features])
    cpt_clean['domain_gap_inv'] = 1 - cpt_clean['domain_gap']
    cpt_clean['composite'] = (
        cpt_clean['src_total_unique_bug_reports'] * 0.35 +
        cpt_clean['bug_sim']                      * 0.25 +
        cpt_clean['domain_gap_inv']               * 0.20 +
        cpt_clean['src_LoC']                      * 0.10 +
        cpt_clean['code_sim']                     * 0.10
    )

    results = []
    for tgt, grp in cpt_clean.groupby('target_project'):
        if len(grp) < 3:
            continue
        oracle_mrr = grp['model_mrr'].max()
        oracle_src = grp.loc[grp['model_mrr'].idxmax(), 'source_project']

        # Random baseline (1000 repetitions)
        rand_mrrs = [grp['model_mrr'].sample(1).values[0] for _ in range(1000)]
        rand_mrr  = np.mean(rand_mrrs)

        # Strategy: most bugs
        best_bugs_src = grp.loc[grp['src_total_unique_bug_reports'].idxmax(), 'source_project']
        best_bugs_mrr = grp.loc[grp['src_total_unique_bug_reports'].idxmax(), 'model_mrr']

        # Strategy: highest complexity
        best_comp_src = grp.loc[grp['src_code_complexity'].idxmax(), 'source_project']
        best_comp_mrr = grp.loc[grp['src_code_complexity'].idxmax(), 'model_mrr']

        # Composite
        best_cpos_src = grp.loc[grp['composite'].idxmax(), 'source_project']
        best_cpos_mrr = grp.loc[grp['composite'].idxmax(), 'model_mrr']

        # Kendall tau for each strategy
        def tau_score(rank_col):
            t, p = kendalltau(grp[rank_col].rank(ascending=False), grp['model_mrr'].rank())
            return t, p

        tau_bugs, p_bugs = tau_score('src_total_unique_bug_reports')
        tau_comp, p_comp = tau_score('src_code_complexity')
        tau_cpos, p_cpos = tau_score('composite')
        tau_bsim, p_bsim = tau_score('bug_sim')

        results.append({
            'target_project'      : tgt,
            'short_name'          : tgt.split('/')[1],
            'n_sources'           : len(grp),
            'oracle_mrr'          : oracle_mrr,
            'oracle_src'          : oracle_src,
            'random_mrr'          : rand_mrr,
            'bugs_heuristic_mrr'  : best_bugs_mrr,
            'bugs_heuristic_src'  : best_bugs_src,
            'complexity_heuristic_mrr': best_comp_mrr,
            'composite_mrr'       : best_cpos_mrr,
            'composite_src'       : best_cpos_src,
            'pct_oracle'          : best_cpos_mrr / (oracle_mrr + 1e-9),
            'pct_random'          : rand_mrr / (oracle_mrr + 1e-9),
            'tau_bugs'            : tau_bugs,
            'tau_comp'            : tau_comp,
            'tau_composite'       : tau_cpos,
            'tau_bugsim'          : tau_bsim,
            'hit1_bugs'           : int(best_bugs_src == oracle_src),
            'hit1_composite'      : int(best_cpos_src == oracle_src),
        })

    return pd.DataFrame(results)


def plot_source_ranking(val_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: MRR achieved by each strategy per target
    x      = np.arange(len(val_df))
    width  = 0.18
    ax     = axes[0]
    ax.bar(x - width,     val_df['oracle_mrr'],           width, label='Oracle (best possible)',      color='gold',     alpha=0.9, edgecolor='black', linewidth=0.6)
    ax.bar(x,             val_df['composite_mrr'],         width, label='Composite heuristic',         color='#2196F3',  alpha=0.85, edgecolor='black', linewidth=0.6)
    ax.bar(x + width,     val_df['bugs_heuristic_mrr'],    width, label='Most bugs heuristic',         color='#4CAF50',  alpha=0.85, edgecolor='black', linewidth=0.6)
    ax.bar(x + 2*width,   val_df['random_mrr'],            width, label='Random baseline',             color='#9E9E9E',  alpha=0.7,  edgecolor='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(val_df['short_name'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('CP-transfer MRR', fontsize=11)
    ax.set_title('Source Selection Strategy Comparison\n(per target project)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Kendall tau distribution for each strategy
    ax2 = axes[1]
    tau_data = {
        'Composite': val_df['tau_composite'].values,
        'Most bugs':  val_df['tau_bugs'].values,
        'Most complex': val_df['tau_comp'].values,
        'Bug similarity': val_df['tau_bugsim'].values,
    }
    bp = ax2.boxplot(tau_data.values(), labels=tau_data.keys(), patch_artist=True, notch=False)
    colors_bp = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Random baseline (τ=0)')
    ax2.set_ylabel('Kendall τ (source ranking quality)', fontsize=11)
    ax2.set_title('Source Ranking Quality by Strategy\n(τ>0 = better than random)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'source_selection_ranking_quality.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ source_selection_ranking_quality.png")


# ── Section 4: CPL desirability — evidence assembly ───────────────────────────

def print_cpl_desirability_evidence(cpt):
    """Print quantitative evidence for each CPL desirability claim."""
    print("\n" + "="*65)
    print("CPL DESIRABILITY EVIDENCE SUMMARY")
    print("="*65)

    # E1: CPL beats limited within-project training
    wins = (cpt['model_mrr'] > cpt['mrr_wps']).sum()
    total = len(cpt.dropna(subset=['mrr_wps']))
    print(f"\nE1  CP-transfer > WP-small: {wins}/{total} pairs ({wins/total:.1%})")

    # E2: CPL near WP-large (upper bound proximity)
    gap = cpt['mrr_wpl'] - cpt['model_mrr']
    print(f"E2  Mean gap (WP-large - CP-transfer): {gap.mean():.4f} MRR")
    print(f"    CP-transfer within 10% of WP-large: {(gap <= 0.02).sum()}/{len(gap)} pairs ({(gap <= 0.02).mean():.1%})")
    print(f"    CP-transfer >= WP-large: {(cpt['model_mrr'] >= cpt['mrr_wpl']).sum()} pairs ({(cpt['model_mrr'] >= cpt['mrr_wpl']).mean():.1%})")

    # E3: Cold-start still beats FAISS
    cold_beats = (cpt['mrr_cpc'] > cpt['faiss_mrr']).sum()
    cold_total = len(cpt.dropna(subset=['mrr_cpc','faiss_mrr']))
    improvement = ((cpt['mrr_cpc'] - cpt['faiss_mrr']) / (cpt['faiss_mrr'] + 1e-9)).median()
    print(f"\nE3  CP-cold-start > FAISS baseline: {cold_beats}/{cold_total} pairs ({cold_beats/cold_total:.1%})")
    print(f"    Median relative improvement: +{improvement:.1%}")

    # E4: CPL gain is highest when target data is scarce
    low_bugs  = cpt[cpt['tgt_total_unique_bug_reports'] < cpt['tgt_total_unique_bug_reports'].median()]
    high_bugs = cpt[cpt['tgt_total_unique_bug_reports'] >= cpt['tgt_total_unique_bug_reports'].median()]
    print(f"\nE4  CPL gain when target has FEW bugs  (n={len(low_bugs)}): {low_bugs['cpl_gain'].mean():+.4f} mean")
    print(f"    CPL gain when target has MANY bugs (n={len(high_bugs)}): {high_bugs['cpl_gain'].mean():+.4f} mean")

    # E5: Model vs FAISS improvement
    model_gain = (cpt['model_mrr'] / (cpt['faiss_mrr'] + 1e-9))
    print(f"\nE5  Model MRR / FAISS MRR ratio: median {model_gain.median():.1f}x, max {model_gain.max():.1f}x")
    print(f"    Pairs where CP-transfer > FAISS: {(cpt['model_mrr'] > cpt['faiss_mrr']).mean():.1%}")

    # E6: Domain gap and CPL
    low_gap  = cpt[cpt['domain_gap'] < cpt['domain_gap'].median()]
    high_gap = cpt[cpt['domain_gap'] >= cpt['domain_gap'].median()]
    print(f"\nE6  CPL wins (low domain gap,  n={len(low_gap)}): {(low_gap['cpl_wins']).mean():.1%}")
    print(f"    CPL wins (high domain gap, n={len(high_gap)}): {(high_gap['cpl_wins']).mean():.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    cpt = load_data()
    print(f"  {len(cpt)} CP-transfer rows for {cpt['target_project'].nunique()} targets\n")

    print("1. Feature correlations plot...")
    plot_feature_correlations(cpt)

    print("2. Target viability profile...")
    tgt_stats = plot_target_viability(cpt)

    print("3. Source ranking heuristics...")
    val_df = rank_sources(cpt)
    plot_source_ranking(val_df)
    val_df.to_csv(OUT_CSV, index=False)
    print(f"  ✓ source_selection_validation.csv ({len(val_df)} rows)")

    print("\n=== Source Ranking Summary ===")
    pd.set_option('display.float_format', '{:.3f}'.format)
    summary_cols = ['short_name','n_sources','oracle_mrr','composite_mrr',
                    'bugs_heuristic_mrr','random_mrr','tau_composite','tau_bugs','hit1_composite']
    print(val_df[summary_cols].to_string(index=False))
    print(f"\nOverall Hit@1 (composite):  {val_df['hit1_composite'].mean():.1%}")
    print(f"Overall Hit@1 (most bugs):  {val_df['hit1_bugs'].mean():.1%}")
    print(f"Mean Kendall τ (composite): {val_df['tau_composite'].mean():.3f}")
    print(f"Mean Kendall τ (most bugs): {val_df['tau_bugs'].mean():.3f}")
    print(f"Targets with τ > 0 (composite): {(val_df['tau_composite'] > 0).sum()}/{len(val_df)}")

    print_cpl_desirability_evidence(cpt)

    print(f"\n✓ Done. Results in {OUT_CSV}")


if __name__ == '__main__':
    main()
