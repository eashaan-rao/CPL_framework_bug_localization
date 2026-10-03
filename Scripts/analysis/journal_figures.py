"""
journal_figures.py
──────────────────
Generates all journal-quality figures for the JSS paper.
Output directory: journal_draft/figs/

Figures produced
────────────────
fig_main_results.png          — Grouped bars: all 5 metrics × 4 scenarios × 3 models
fig_win_rate_metrics.png      — CPT vs WPS win rate by model × metric
fig_cohens_d_all_metrics.png  — Cohen's d heatmap: all 5 metrics × comparisons × 3 models
fig_gain_distribution.png     — CPL gain histograms: MRR + MAP per model (2-row)
fig_gain_by_size.png          — Violin: MRR + MAP CPL gain by target size × model
fig_cross_model_scatter.png   — 3-panel: BLAZE×COOBA, BLAZE×TRANP-CNN, COOBA×TRANP-CNN
fig_coldstart_metrics.png     — Cold-start MRR / MAP / Top-5 vs target LoC
fig_neg_transfer_mrr_map.png  — MRR vs MAP delta scatter (negative transfer identity)

Run
───
    python Scripts/analysis/journal_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import wilcoxon, spearmanr

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = "/home/cs21d002_eashaan/PhD/Objective1"
RESULTS_CSV  = f"{ROOT}/results/paper_results_complete_corrected.csv"
METADATA_PKL = f"{ROOT}/data/processed/project_metadata.parquet"
EFFECT_CSV   = f"{ROOT}/results/effect_size_summary.csv"
OUT_DIR      = f"{ROOT}/journal_draft/figs"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared style ───────────────────────────────────────────────────────────────
MODEL_COLOR  = {'BLAZE': '#1565C0', 'COOBA': '#E65100', 'TRANP-CNN': '#6A1B9A'}
MODEL_MARKER = {'BLAZE': 'o',       'COOBA': 's',        'TRANP-CNN': '^'}
MODEL_LABEL  = {'BLAZE': 'BLAZE',   'COOBA': 'COOBA',    'TRANP-CNN': 'TRANP-CNN'}

SCENARIO_COLOR = {
    'WP-small':      '#90CAF9',
    'WP-large':      '#1565C0',
    'CP-cold-start': '#B0BEC5',
    'CP-transfer':   '#E65100',
}
SCENARIO_LABEL = {
    'WP-small':      'WP-small',
    'WP-large':      'WP-large',
    'CP-cold-start': 'CP-cold-start',
    'CP-transfer':   'CP-transfer',
}
SCENARIOS = ['WP-small', 'WP-large', 'CP-cold-start', 'CP-transfer']
MODELS    = ['BLAZE', 'COOBA', 'TRANP-CNN']
METRICS   = ['MRR', 'MAP', 'top-1', 'top-5', 'top-10']
METRIC_LABEL = {'MRR': 'MRR', 'MAP': 'MAP', 'top-1': 'Top-1', 'top-5': 'Top-5', 'top-10': 'Top-10'}

POS_THRESH =  0.01
NEG_THRESH = -0.01

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.linestyle':     '--',
})

DPI = 200

# ── Data loading ───────────────────────────────────────────────────────────────
def load():
    df = pd.read_csv(RESULTS_CSV)
    meta = pd.read_parquet(METADATA_PKL)[['repo_name', 'LoC', 'total_unique_bug_reports', 'domain']]
    return df, meta

def pivot_cpt_wps(df, metric='MRR'):
    """Returns per-model DataFrame with columns delta_{metric}, source, target."""
    rows = []
    for model, mg in df.groupby('model_name'):
        cpt = mg[mg['scenario'] == 'CP-transfer'].set_index(['source_project', 'target_project'])[metric]
        wps = mg[mg['scenario'] == 'WP-small'].set_index(['source_project', 'target_project'])[metric]
        idx = cpt.index.intersection(wps.index)
        for (src, tgt) in idx:
            rows.append({'model': model, 'source': src, 'target': tgt,
                         'delta': cpt[(src, tgt)] - wps[(src, tgt)],
                         'cpt': cpt[(src, tgt)], 'wps': wps[(src, tgt)]})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Main results: grouped bar, all 5 metrics × 4 scenarios × 3 models
# ══════════════════════════════════════════════════════════════════════════════
def fig_main_results(df):
    means = {}
    for model, mg in df.groupby('model_name'):
        for scen, sg in mg.groupby('scenario'):
            for metric in METRICS:
                means[(model, scen, metric)] = sg[metric].mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    x     = np.arange(len(METRICS))
    n_sc  = len(SCENARIOS)
    w     = 0.18
    offsets = np.linspace(-(n_sc - 1) / 2 * w, (n_sc - 1) / 2 * w, n_sc)

    for ax, model in zip(axes, MODELS):
        for i, (scen, off) in enumerate(zip(SCENARIOS, offsets)):
            vals = [means.get((model, scen, m), 0) for m in METRICS]
            bars = ax.bar(x + off, vals, w,
                          label=SCENARIO_LABEL[scen],
                          color=SCENARIO_COLOR[scen],
                          edgecolor='white', linewidth=0.5, alpha=0.92)
            for bar, val in zip(bars, vals):
                if val > 0.02:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=6.5, rotation=90)

        ax.set_ylim(0, 0.80)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{model}', fontsize=13, fontweight='bold',
                     color=MODEL_COLOR[model])
        ax.tick_params(axis='y', labelsize=9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([METRIC_LABEL[m] for m in METRICS], fontsize=12)
    axes[-1].set_xlabel('Metric', fontsize=12)

    handles = [mpatches.Patch(color=SCENARIO_COLOR[s], label=SCENARIO_LABEL[s]) for s in SCENARIOS]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle('Bug localization performance across scenarios and metrics\n(n = 63 project pairs)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_main_results.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Win rate per model × metric (CPT vs WPS)
# ══════════════════════════════════════════════════════════════════════════════
def fig_win_rate_metrics(df):
    rows = []
    for model, mg in df.groupby('model_name'):
        cpt = mg[mg['scenario'] == 'CP-transfer'].set_index(['source_project', 'target_project'])
        wps = mg[mg['scenario'] == 'WP-small'].set_index(['source_project', 'target_project'])
        idx = cpt.index.intersection(wps.index)
        n = len(idx)
        for metric in METRICS:
            wins = (cpt.loc[idx, metric].values > wps.loc[idx, metric].values).sum()
            rows.append({'model': model, 'metric': metric, 'win_rate': wins / n, 'n': n})
    wr = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(METRICS))
    w = 0.24
    offsets = np.linspace(-(len(MODELS) - 1) / 2 * w, (len(MODELS) - 1) / 2 * w, len(MODELS))

    for model, off in zip(MODELS, offsets):
        vals = [wr[(wr['model'] == model) & (wr['metric'] == m)]['win_rate'].values[0]
                for m in METRICS]
        bars = ax.bar(x + off, vals, w, label=model,
                      color=MODEL_COLOR[model], edgecolor='white', linewidth=0.4, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=8)

    ax.axhline(0.5, color='gray', linewidth=1.0, linestyle='--', alpha=0.7, label='50% baseline')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[m] for m in METRICS], fontsize=11)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Win rate (CPT > WPS)', fontsize=12)
    ax.set_title('CP-transfer win rate over WP-small by metric and model\n(n = 63 pairs each)',
                 fontsize=12, fontweight='bold')
    handles = [mpatches.Patch(color=MODEL_COLOR[m], label=m) for m in MODELS]
    handles.append(plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.0, label='50% baseline'))
    ax.legend(handles=handles, fontsize=9, loc='lower right')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_win_rate_metrics.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Cohen's d heatmap: all 5 metrics × comparisons × 3 models
# ══════════════════════════════════════════════════════════════════════════════
def fig_cohens_d_all_metrics(df):
    COMPARISONS = [
        ('CP-transfer',   'WP-small',    'CPT vs WPS'),
        ('CP-transfer',   'WP-large',    'CPT vs WPL'),
        ('CP-cold-start', 'WP-small',    'CPC vs WPS'),
        ('WP-large',      'WP-small',    'WPL vs WPS'),
        ('CP-transfer',   'CP-cold-start', 'CPT vs CPC'),
    ]

    def cohens_d(a, b):
        diff = a - b
        pooled_sd = np.std(diff, ddof=1)
        return diff.mean() / pooled_sd if pooled_sd > 0 else 0.0

    def sig_label(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return ''

    records = []
    for model, mg in df.groupby('model_name'):
        for sc_a, sc_b, label in COMPARISONS:
            ga = mg[mg['scenario'] == sc_a].set_index(['source_project', 'target_project'])
            gb = mg[mg['scenario'] == sc_b].set_index(['source_project', 'target_project'])
            idx = ga.index.intersection(gb.index)
            for metric in METRICS:
                a = ga.loc[idx, metric].values
                b = gb.loc[idx, metric].values
                d = cohens_d(a, b)
                try:
                    _, p = wilcoxon(a, b, alternative='greater')
                except Exception:
                    p = 1.0
                records.append({'model': model, 'comparison': label,
                                 'metric': metric, 'd': d, 'p': p,
                                 'sig': sig_label(p)})
    eff = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    for ax, model in zip(axes, MODELS):
        sub = eff[eff['model'] == model]
        pivot_d   = sub.pivot(index='comparison', columns='metric', values='d')[METRICS]
        pivot_sig = sub.pivot(index='comparison', columns='metric', values='sig')[METRICS]

        comp_order = [c[2] for c in COMPARISONS]
        pivot_d   = pivot_d.reindex(comp_order)
        pivot_sig = pivot_sig.reindex(comp_order)

        sns.heatmap(pivot_d, annot=False, cmap=cmap,
                    vmin=-1.5, vmax=1.5, ax=ax, linewidths=0.5,
                    cbar_kws={'label': "Cohen's d"})

        for i, row_label in enumerate(comp_order):
            for j, col_label in enumerate(METRICS):
                d_val = pivot_d.loc[row_label, col_label]
                sig   = pivot_sig.loc[row_label, col_label]
                cell_text = f'{d_val:.2f}{sig}'
                text_color = 'white' if abs(d_val) > 0.7 else 'black'
                ax.text(j + 0.5, i + 0.5, cell_text,
                        ha='center', va='center', fontsize=8.5,
                        color=text_color, fontweight='bold' if sig else 'normal')

        ax.set_title(f'{model}', fontsize=13, fontweight='bold', color=MODEL_COLOR[model])
        ax.set_xlabel('Metric', fontsize=10)
        ax.set_ylabel('Comparison' if model == 'BLAZE' else '', fontsize=10)
        ax.set_xticklabels([METRIC_LABEL[m] for m in METRICS], fontsize=9)
        ax.set_yticklabels(comp_order, fontsize=9, rotation=0)

    fig.suptitle("Cohen's d for scenario comparisons (all metrics)\n* p<0.05  ** p<0.01  *** p<0.001",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_cohens_d_all_metrics.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — CPL gain distributions: MRR + MAP (2-row × 3-model)
# ══════════════════════════════════════════════════════════════════════════════
def fig_gain_distribution(df):
    plot_metrics = [('MRR', 'MRR'), ('MAP', 'MAP')]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey='row')

    for row_idx, (metric, mlabel) in enumerate(plot_metrics):
        delta_data = pivot_cpt_wps(df, metric)
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]
            sub = delta_data[delta_data['model'] == model]['delta'].dropna()
            color = MODEL_COLOR[model]

            ax.hist(sub, bins=16, color=color, edgecolor='white', alpha=0.82, linewidth=0.5)
            ax.axvline(0,           color='#E53935', linewidth=1.5, linestyle='--', label='Zero')
            ax.axvline(sub.mean(),  color='#FB8C00', linewidth=1.5, linestyle='-',
                       label=f'Mean {sub.mean():+.3f}')
            ax.axvline(sub.median(), color='#43A047', linewidth=1.5, linestyle='-.',
                       label=f'Median {sub.median():+.3f}')
            ax.axvline(NEG_THRESH,  color='#E53935', linewidth=0.7, linestyle=':')
            ax.axvline(POS_THRESH,  color='#43A047', linewidth=0.7, linestyle=':')

            n_neg = (sub < NEG_THRESH).sum()
            n_pos = (sub > POS_THRESH).sum()
            ax.text(0.97, 0.97,
                    f'Win: {(sub > 0).sum()}/{len(sub)}\nNeg: {n_neg}  Pos: {n_pos}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

            if row_idx == 0:
                ax.set_title(model, fontsize=12, fontweight='bold', color=color)
            ax.set_xlabel(f'CPT − WPS  ({mlabel})', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel('Number of pairs', fontsize=10)
            ax.legend(fontsize=7.5, loc='upper left')

    fig.suptitle('Distribution of CPL gain (CP-transfer − WP-small) across 63 project pairs',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_gain_distribution.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Violin: MRR + MAP gain by target size × model
# ══════════════════════════════════════════════════════════════════════════════
def fig_gain_by_size(df, meta):
    tgt_meta = meta.rename(columns={'repo_name': 'target_project',
                                    'LoC': 'tgt_LoC'})
    bins   = [0, 100_000, 300_000, float('inf')]
    labels = ['Small\n(<100K)', 'Medium\n(100K–300K)', 'Large\n(>300K)']

    rows = []
    for metric in ['MRR', 'MAP']:
        delta = pivot_cpt_wps(df, metric)
        merged = delta.merge(tgt_meta[['target_project', 'tgt_LoC']],
                             left_on='target', right_on='target_project', how='left')
        merged['size_group'] = pd.cut(merged['tgt_LoC'], bins=bins, labels=labels)
        merged['metric'] = metric
        rows.append(merged)
    all_delta = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, metric in zip(axes, ['MRR', 'MAP']):
        sub = all_delta[all_delta['metric'] == metric]
        palette = {m: MODEL_COLOR[m] for m in MODELS}
        sns.violinplot(data=sub, x='size_group', y='delta', hue='model',
                       split=False, inner='box', palette=palette,
                       ax=ax, alpha=0.78, order=labels, hue_order=MODELS,
                       linewidth=0.7)
        ax.axhline(0, color='#E53935', linewidth=1.3, linestyle='--', alpha=0.85, label='Zero gain')
        ax.set_xlabel('Target codebase size', fontsize=11)
        ax.set_ylabel(f'CPT − WPS ({metric})', fontsize=11)
        ax.set_title(f'CPL gain by target size — {metric}', fontsize=12, fontweight='bold')
        ax.legend(title='Model', fontsize=9, title_fontsize=9)
        ax.tick_params(axis='x', labelsize=10)

    fig.suptitle('Effect of target codebase size on CPL gain\n(CP-transfer − WP-small, n = 63 pairs)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_gain_by_size.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Cross-model scatter: 3 pairs × MRR delta
# ══════════════════════════════════════════════════════════════════════════════
def fig_cross_model_scatter(df):
    delta_mrr = {}
    for model, mg in df.groupby('model_name'):
        cpt = mg[mg['scenario'] == 'CP-transfer'].set_index(['source_project', 'target_project'])['MRR']
        wps = mg[mg['scenario'] == 'WP-small'].set_index(['source_project', 'target_project'])['MRR']
        idx = cpt.index.intersection(wps.index)
        delta_mrr[model] = (cpt[idx] - wps[idx]).rename(model)

    pairs = [('BLAZE', 'COOBA'), ('BLAZE', 'TRANP-CNN'), ('COOBA', 'TRANP-CNN')]

    def direction_color(d1, d2):
        sign1 = 1 if d1 > POS_THRESH else (-1 if d1 < NEG_THRESH else 0)
        sign2 = 1 if d2 > POS_THRESH else (-1 if d2 < NEG_THRESH else 0)
        if sign1 == sign2 and sign1 != 0:
            return '#4CAF50'  # both agree
        if sign1 == -1 or sign2 == -1:
            return '#F44336'  # at least one negative
        return '#FFC107'      # neutral / mixed

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (m1, m2) in zip(axes, pairs):
        idx = delta_mrr[m1].index.intersection(delta_mrr[m2].index)
        x = delta_mrr[m1][idx].values
        y = delta_mrr[m2][idx].values
        colors = [direction_color(xi, yi) for xi, yi in zip(x, y)]
        rho, p = spearmanr(x, y)

        ax.scatter(x, y, c=colors, s=55, edgecolors='k', linewidth=0.4, alpha=0.85, zorder=3)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)

        lim = max(abs(x).max(), abs(y).max()) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        diag = np.array([-lim, lim])
        ax.plot(diag, diag, color='#BDBDBD', linewidth=0.7, linestyle=':', zorder=1)

        sig_str = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax.text(0.04, 0.96, f'ρ = {rho:.3f} ({sig_str})\nn = {len(idx)}',
                transform=ax.transAxes, va='top', fontsize=9.5,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.set_xlabel(f'{m1} CPL gain (MRR)', fontsize=10)
        ax.set_ylabel(f'{m2} CPL gain (MRR)', fontsize=10)
        ax.set_title(f'{m1}  vs  {m2}', fontsize=12, fontweight='bold')

    agree_patch  = mpatches.Patch(color='#4CAF50', label='Both agree (positive)')
    neg_patch    = mpatches.Patch(color='#F44336', label='Negative transfer present')
    mixed_patch  = mpatches.Patch(color='#FFC107', label='Neutral / mixed')
    fig.legend(handles=[agree_patch, neg_patch, mixed_patch],
               loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Cross-model CPL gain agreement (CP-transfer − WP-small, MRR)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_cross_model_scatter.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Cold-start: MRR / MAP / Top-5 vs target LoC
# ══════════════════════════════════════════════════════════════════════════════
def fig_coldstart_metrics(df, meta):
    VIABLE_MRR = 0.20
    VIABLE_MAP = 0.15
    cpc = df[df['scenario'] == 'CP-cold-start'].copy()
    tgt_meta = meta.rename(columns={'repo_name': 'target_project', 'LoC': 'tgt_LoC'})
    cpc = cpc.merge(tgt_meta[['target_project', 'tgt_LoC']], on='target_project', how='left')

    plot_metrics = [
        ('MRR',   'MRR',   VIABLE_MRR, '#1565C0'),
        ('MAP',   'MAP',   VIABLE_MAP, '#E65100'),
        ('top-5', 'Top-5', 0.35,       '#6A1B9A'),
    ]

    fig, axes = plt.subplots(len(plot_metrics), len(MODELS),
                             figsize=(14, 4 * len(plot_metrics)), sharey='row', sharex='col')

    for row_idx, (metric, mlabel, thresh, col) in enumerate(plot_metrics):
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]
            sub = cpc[cpc['model_name'] == model]
            viable = sub[metric] >= thresh
            ax.scatter(sub.loc[~viable, 'tgt_LoC'] / 1_000, sub.loc[~viable, metric],
                       color='#90A4AE', s=45, alpha=0.8, edgecolors='k',
                       linewidth=0.4, label='Not viable', zorder=3)
            ax.scatter(sub.loc[viable, 'tgt_LoC'] / 1_000, sub.loc[viable, metric],
                       color=col, s=55, alpha=0.9, edgecolors='k',
                       linewidth=0.4, label=f'Viable (≥ {thresh})', zorder=4)
            ax.axhline(thresh, color=col, linewidth=1.2, linestyle='--', alpha=0.75)

            n_viable = viable.sum()
            ax.text(0.97, 0.97, f'{n_viable}/{len(sub)} viable',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            if row_idx == 0:
                ax.set_title(model, fontsize=12, fontweight='bold',
                             color=MODEL_COLOR[model])
            if col_idx == 0:
                ax.set_ylabel(f'CP-cold-start {mlabel}', fontsize=10)
            if row_idx == len(plot_metrics) - 1:
                ax.set_xlabel('Target LoC (thousands)', fontsize=10)
            ax.legend(fontsize=8, loc='lower right')

    fig.suptitle('Zero-shot (CP-cold-start) performance vs target codebase size',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_coldstart_metrics.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — MRR vs MAP delta scatter (negative transfer concordance)
# ══════════════════════════════════════════════════════════════════════════════
def fig_neg_transfer_mrr_map(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, model in zip(axes, MODELS):
        mg = df[df['model_name'] == model]
        cpt = mg[mg['scenario'] == 'CP-transfer'].set_index(['source_project', 'target_project'])
        wps = mg[mg['scenario'] == 'WP-small'].set_index(['source_project', 'target_project'])
        idx = cpt.index.intersection(wps.index)
        delta_mrr = (cpt.loc[idx, 'MRR'] - wps.loc[idx, 'MRR']).values
        delta_map = (cpt.loc[idx, 'MAP'] - wps.loc[idx, 'MAP']).values

        def pt_color(dm, dp):
            if dm < NEG_THRESH or dp < NEG_THRESH:
                return '#F44336'
            if dm > POS_THRESH and dp > POS_THRESH:
                return '#4CAF50'
            return '#FFC107'

        colors = [pt_color(dm, dp) for dm, dp in zip(delta_mrr, delta_map)]
        ax.scatter(delta_mrr, delta_map, c=colors, s=55,
                   edgecolors='k', linewidth=0.4, alpha=0.85, zorder=3)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(NEG_THRESH, color='#F44336', linewidth=0.6, linestyle=':')
        ax.axvline(NEG_THRESH, color='#F44336', linewidth=0.6, linestyle=':')
        ax.axhline(POS_THRESH, color='#4CAF50', linewidth=0.6, linestyle=':')
        ax.axvline(POS_THRESH, color='#4CAF50', linewidth=0.6, linestyle=':')

        rho, p = spearmanr(delta_mrr, delta_map)
        sig_str = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        n_neg = sum(dm < NEG_THRESH or dp < NEG_THRESH for dm, dp in zip(delta_mrr, delta_map))
        ax.text(0.04, 0.96,
                f'ρ = {rho:.3f} ({sig_str})\nNeg. cases: {n_neg}/{len(idx)}',
                transform=ax.transAxes, va='top', fontsize=9.5,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.set_xlabel('CPT − WPS  (MRR)', fontsize=10)
        ax.set_ylabel('CPT − WPS  (MAP)', fontsize=10)
        ax.set_title(model, fontsize=12, fontweight='bold', color=MODEL_COLOR[model])

    agree_patch = mpatches.Patch(color='#4CAF50', label='Both positive')
    neg_patch   = mpatches.Patch(color='#F44336', label='Negative in MRR or MAP')
    mixed_patch = mpatches.Patch(color='#FFC107', label='Neutral / mixed')
    fig.legend(handles=[agree_patch, neg_patch, mixed_patch],
               loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('MRR vs MAP CPL gain concordance (CP-transfer − WP-small)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_neg_transfer_mrr_map.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Loading data...')
    df, meta = load()
    print(f'  {len(df)} rows, models: {sorted(df["model_name"].unique())}')
    print()

    print('Figure 1: Main results...')
    fig_main_results(df)

    print('Figure 2: Win rate by metric...')
    fig_win_rate_metrics(df)

    print("Figure 3: Cohen's d heatmap (all metrics)...")
    fig_cohens_d_all_metrics(df)

    print('Figure 4: Gain distributions (MRR + MAP)...')
    fig_gain_distribution(df)

    print('Figure 5: Gain by target size (MRR + MAP)...')
    fig_gain_by_size(df, meta)

    print('Figure 6: Cross-model scatter (3 pairs)...')
    fig_cross_model_scatter(df)

    print('Figure 7: Cold-start by metric...')
    fig_coldstart_metrics(df, meta)

    print('Figure 8: MRR vs MAP delta (negative transfer)...')
    fig_neg_transfer_mrr_map(df)

    print()
    print(f'All figures saved to {OUT_DIR}')
