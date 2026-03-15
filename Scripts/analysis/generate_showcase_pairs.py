"""
generate_showcase_pairs.py
──────────────────────────
Generates focused plots and CSV summaries for three showcase project pairs
that illustrate contrasting CPL behaviours.

Pair A — matplotlib ↔ jupyterlab  (commutativity contrast: CPL helps one way, hurts the other)
Pair B — numpy → jupyterlab        (best overall performer, strong CPL win)
Pair C — numpy → scipy             (worst performer: large-codebase ceiling)

Output
──────
results/showcase_pairs/
    pair_A_matplotlib_jupyterlab/
    pair_B_numpy_jupyterlab/
    pair_C_numpy_scipy/

Run
───
    python Scripts/analysis/generate_showcase_pairs.py
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
SUMMARY_CSV   = "/home/cs21d002_eashaan/PhD/Objective1/results/tranp_cnn_ph1_summary.csv"
DIAG_DIR      = "/home/cs21d002_eashaan/PhD/Objective1/results/tranp_cnn_ph1_diagnostics"
SHOWCASE_DIR  = "/home/cs21d002_eashaan/PhD/Objective1/results/showcase_pairs"

SCENARIOS      = ['WP-small', 'WP-large', 'CP-cold-start', 'CP-transfer']
SCENARIO_COLORS = {
    'WP-small'    : '#2196F3',
    'WP-large'    : '#4CAF50',
    'CP-cold-start': '#FF9800',
    'CP-transfer' : '#E91E63',
}

SHOWCASE_PAIRS = {
    'pair_A_matplotlib_jupyterlab': [
        ('matplotlib/matplotlib', 'jupyterlab/jupyterlab'),
        ('jupyterlab/jupyterlab', 'matplotlib/matplotlib'),
    ],
    'pair_B_numpy_jupyterlab': [
        ('numpy/numpy',             'jupyterlab/jupyterlab'),
        ('jupyterlab/jupyterlab',   'numpy/numpy'),
    ],
    'pair_C_numpy_scipy': [
        ('numpy/numpy',   'scipy/scipy'),
        ('scipy/scipy',   'numpy/numpy'),
    ],
}

PAIR_TITLES = {
    'pair_A_matplotlib_jupyterlab': 'Pair A: matplotlib ↔ jupyterlab\n(Commutativity Contrast — CPL helps one direction, hurts the other)',
    'pair_B_numpy_jupyterlab'     : 'Pair B: numpy ↔ jupyterlab\n(Best Performer — Strong CPL Signal)',
    'pair_C_numpy_scipy'          : 'Pair C: numpy ↔ scipy\n(Hard Target — Large Codebase Ceiling)',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_name(repo):
    return repo.replace('/', '_')


def copy_diagnostics(src_repo, tgt_repo, dest_dir):
    """Copy all 4 scenario diagnostic CSVs for a given pair into dest_dir."""
    for scenario in SCENARIOS:
        fname = f"{safe_name(src_repo)}_{safe_name(tgt_repo)}_{scenario}_diagnostics.csv"
        src_path = os.path.join(DIAG_DIR, fname)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(dest_dir, fname))


def get_pair_data(df, src, tgt):
    subset = df[(df['source_project'] == src) & (df['target_project'] == tgt)].copy()
    subset = subset.set_index('scenario').reindex(SCENARIOS)
    return subset


def plot_scenario_bar(ax, data_row, src, tgt, title_suffix=''):
    """
    Grouped bar chart: model MRR vs FAISS MRR per scenario.
    """
    x      = np.arange(len(SCENARIOS))
    width  = 0.35
    colors = [SCENARIO_COLORS[s] for s in SCENARIOS]

    model_vals = [data_row.loc[s, 'model_mrr'] if s in data_row.index else 0 for s in SCENARIOS]
    faiss_vals = [data_row.loc[s, 'faiss_mrr'] if s in data_row.index else 0 for s in SCENARIOS]

    bars1 = ax.bar(x - width/2, model_vals, width, label='TRANP-CNN (model)', color=colors, alpha=0.9, edgecolor='black', linewidth=0.6)
    bars2 = ax.bar(x + width/2, faiss_vals, width, label='FAISS baseline', color=colors, alpha=0.35, edgecolor='black', linewidth=0.6, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, fontsize=9)
    ax.set_ylabel('MRR', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{src.split("/")[1]} → {tgt.split("/")[1]}  {title_suffix}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Annotate bars with values
    for bar, v in zip(bars1, model_vals):
        if v > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7)


def plot_top_k_comparison(ax, data_row, src, tgt):
    """
    Line chart: Top-1, Top-5, Top-10 per scenario for model vs FAISS.
    """
    model_top = {
        'Top-1' : [data_row.loc[s, 'model_top1']  if s in data_row.index else 0 for s in SCENARIOS],
        'Top-5' : [data_row.loc[s, 'model_top5']  if s in data_row.index else 0 for s in SCENARIOS],
        'Top-10': [data_row.loc[s, 'model_top10'] if s in data_row.index else 0 for s in SCENARIOS],
    }
    x = np.arange(len(SCENARIOS))
    styles = {'Top-1': 'o-', 'Top-5': 's--', 'Top-10': '^:'}
    colors_line = {'Top-1': '#E91E63', 'Top-5': '#9C27B0', 'Top-10': '#3F51B5'}
    for k, vals in model_top.items():
        ax.plot(x, vals, styles[k], color=colors_line[k], label=k, linewidth=2, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, fontsize=9)
    ax.set_ylabel('Recall', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title(f'{src.split("/")[1]} → {tgt.split("/")[1]}  (Recall@K)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_rank_displacement(ax, diag_data, src, tgt, scenario):
    """Box plot of rank displacement for a given scenario."""
    if diag_data is None or diag_data.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    color = SCENARIO_COLORS.get(scenario, 'steelblue')
    ax.boxplot(diag_data['rank_displacement'], vert=True, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.6),
               medianprops=dict(color='black', linewidth=2))
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='No change')
    ax.set_ylabel('Rank displacement\n(+ve = model improved)', fontsize=9)
    ax.set_title(f'{scenario}\n{src.split("/")[1]}→{tgt.split("/")[1]}', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_pair_showcase(pair_key, directions, df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Copy diagnostic CSVs
    for src, tgt in directions:
        copy_diagnostics(src, tgt, out_dir)

    # 2. Save per-pair summary CSV
    all_rows = []
    for src, tgt in directions:
        rows = df[(df['source_project'] == src) & (df['target_project'] == tgt)].copy()
        all_rows.append(rows)
    pair_summary = pd.concat(all_rows)
    pair_summary.to_csv(os.path.join(out_dir, 'pair_summary.csv'), index=False)

    # 3. Main comparison figure: MRR + Recall@K for both directions
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(PAIR_TITLES[pair_key], fontsize=12, fontweight='bold', y=1.01)

    for col_idx, (src, tgt) in enumerate(directions):
        data = get_pair_data(df, src, tgt)
        plot_scenario_bar(axes[0, col_idx], data, src, tgt)
        plot_top_k_comparison(axes[1, col_idx], data, src, tgt)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'mrr_and_recall_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ mrr_and_recall_comparison.png")

    # 4. Rank displacement per scenario for first direction
    src0, tgt0 = directions[0]
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
    fig2.suptitle(f'Rank Displacement — {src0.split("/")[1]} → {tgt0.split("/")[1]}\n(positive = model ranked GT higher than FAISS)',
                  fontsize=11, fontweight='bold')
    for i, scenario in enumerate(SCENARIOS):
        fname = f"{safe_name(src0)}_{safe_name(tgt0)}_{scenario}_diagnostics.csv"
        fpath = os.path.join(DIAG_DIR, fname)
        diag  = pd.read_csv(fpath) if os.path.exists(fpath) else None
        plot_rank_displacement(axes2[i], diag, src0, tgt0, scenario)
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'rank_displacement_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ rank_displacement_boxplot.png")

    # 5. Commutativity comparison bar (Pair A only — both directions side by side)
    if len(directions) == 2:
        src0, tgt0 = directions[0]
        src1, tgt1 = directions[1]
        d0 = get_pair_data(df, src0, tgt0)
        d1 = get_pair_data(df, src1, tgt1)

        x = np.arange(len(SCENARIOS))
        w = 0.3
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        colors = [SCENARIO_COLORS[s] for s in SCENARIOS]
        mrr0 = [d0.loc[s, 'model_mrr'] if s in d0.index else 0 for s in SCENARIOS]
        mrr1 = [d1.loc[s, 'model_mrr'] if s in d1.index else 0 for s in SCENARIOS]

        b0 = ax3.bar(x - w/2, mrr0, w, label=f'{src0.split("/")[1]}→{tgt0.split("/")[1]}',
                     color='#2196F3', alpha=0.85, edgecolor='black', linewidth=0.7)
        b1 = ax3.bar(x + w/2, mrr1, w, label=f'{src1.split("/")[1]}→{tgt1.split("/")[1]}',
                     color='#E91E63', alpha=0.85, edgecolor='black', linewidth=0.7)

        ax3.set_xticks(x)
        ax3.set_xticklabels(SCENARIOS, fontsize=10)
        ax3.set_ylabel('Model MRR', fontsize=11)
        ax3.set_ylim(0, 1.0)
        ax3.set_title(f'Commutativity: {src0.split("/")[1]} ↔ {tgt0.split("/")[1]}\nSame domain gap ({d0.iloc[0]["domain_gap"]:.3f}), same projects — different direction = different performance',
                      fontsize=11, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3)

        for b, v in zip(b0, mrr0):
            if v > 0.01:
                ax3.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        for b, v in zip(b1, mrr1):
            if v > 0.01:
                ax3.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        fig3.savefig(os.path.join(out_dir, 'commutativity_contrast.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ commutativity_contrast.png")

    print(f"  ✓ pair_summary.csv + {len(directions)*4} diagnostic CSVs copied")


def main():
    df = pd.read_csv(SUMMARY_CSV)
    os.makedirs(SHOWCASE_DIR, exist_ok=True)

    for pair_key, directions in SHOWCASE_PAIRS.items():
        out_dir = os.path.join(SHOWCASE_DIR, pair_key)
        print(f"\n── {pair_key} ──")
        generate_pair_showcase(pair_key, directions, df, out_dir)

    print(f"\n✓ All showcase pairs written to {SHOWCASE_DIR}/")


if __name__ == '__main__':
    main()
