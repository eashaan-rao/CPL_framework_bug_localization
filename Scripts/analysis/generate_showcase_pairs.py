"""
generate_showcase_pairs.py
──────────────────────────
Generates focused plots and CSV summaries for three showcase project pairs
that illustrate contrasting CPL behaviours.

Pair A — matplotlib ↔ jupyterlab  (commutativity contrast: CPL helps one way, hurts the other)
Pair B — numpy → jupyterlab        (best overall performer, strong CPL win)
Pair C — numpy → scipy             (worst performer: large-codebase ceiling)

Metrics reported: Top-1, Top-5, Top-10, MAP, MRR

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

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_CSV   = "/home/cs21d002_eashaan/PhD/Objective1/results/phase1_experimental_results.csv"
SUMMARY_CSV   = "/home/cs21d002_eashaan/PhD/Objective1/results/tranp_cnn_ph1_summary.csv"   # for FAISS MRR only
DIAG_DIR      = "/home/cs21d002_eashaan/PhD/Objective1/results/tranp_cnn_ph1_diagnostics"
SHOWCASE_DIR  = "/home/cs21d002_eashaan/PhD/Objective1/results/showcase_pairs"

SCENARIOS = ['WP-small', 'WP-large', 'CP-cold-start', 'CP-transfer']
SCENARIO_COLORS = {
    'WP-small'     : '#2196F3',
    'WP-large'     : '#4CAF50',
    'CP-cold-start': '#FF9800',
    'CP-transfer'  : '#E91E63',
}

SHOWCASE_PAIRS = {
    'pair_A_matplotlib_jupyterlab': [
        ('matplotlib/matplotlib',   'jupyterlab/jupyterlab'),
        ('jupyterlab/jupyterlab',   'matplotlib/matplotlib'),
    ],
    'pair_B_numpy_jupyterlab': [
        ('numpy/numpy',             'jupyterlab/jupyterlab'),
        ('jupyterlab/jupyterlab',   'numpy/numpy'),
    ],
    'pair_C_numpy_scipy': [
        ('numpy/numpy',             'scipy/scipy'),
        ('scipy/scipy',             'numpy/numpy'),
    ],
}

PAIR_TITLES = {
    'pair_A_matplotlib_jupyterlab': 'Pair A: matplotlib ↔ jupyterlab  (Commutativity Contrast)',
    'pair_B_numpy_jupyterlab'     : 'Pair B: numpy ↔ jupyterlab  (Best CPL Performer)',
    'pair_C_numpy_scipy'          : 'Pair C: numpy ↔ scipy  (Hard Target — Large Codebase Ceiling)',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_name(repo):
    return repo.replace('/', '_')


def copy_diagnostics(src_repo, tgt_repo, dest_dir):
    for scenario in SCENARIOS:
        fname = f"{safe_name(src_repo)}_{safe_name(tgt_repo)}_{scenario}_diagnostics.csv"
        src_path = os.path.join(DIAG_DIR, fname)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(dest_dir, fname))


def get_pair_rows(df, src, tgt):
    """Return scenario-indexed DataFrame for a given src→tgt pair."""
    subset = df[(df['source_project'] == src) & (df['target_project'] == tgt)].copy()
    subset = subset.set_index('scenario').reindex(SCENARIOS)
    return subset


def val(df, scenario, col, default=0.0):
    """Safe accessor: df is scenario-indexed, returns df.loc[scenario, col]."""
    try:
        v = df.loc[scenario, col]
        return float(v) if pd.notna(v) else default
    except Exception:
        return default


# ── Plot 1: Full metrics comparison (3 rows × 2 cols) ─────────────────────────

def plot_full_metrics(pair_key, directions, df, faiss_df, out_dir):
    """
    3-row × 2-col figure:
      Row 0: MRR per scenario (both directions)
      Row 1: MAP per scenario (both directions)
      Row 2: Top-K recall lines (both directions)
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(PAIR_TITLES[pair_key], fontsize=13, fontweight='bold', y=1.01)

    for col_idx, (src, tgt) in enumerate(directions):
        data = get_pair_rows(df, src, tgt)
        # Build per-direction FAISS df (scenario-indexed)
        if faiss_df is not None and not faiss_df.empty:
            fraw = faiss_df[(faiss_df['source_project'] == src) &
                            (faiss_df['target_project'] == tgt)].copy()
            fdata_dir = fraw.set_index('scenario').reindex(SCENARIOS) if len(fraw) else pd.DataFrame()
        else:
            fdata_dir = pd.DataFrame()
        x      = np.arange(len(SCENARIOS))
        w      = 0.35
        colors = [SCENARIO_COLORS[s] for s in SCENARIOS]
        src_lbl = src.split('/')[1]
        tgt_lbl = tgt.split('/')[1]

        # ── Row 0: MRR ──
        ax = axes[0, col_idx]
        mrr_model = [val(data, s, 'MRR') for s in SCENARIOS]
        mrr_faiss = [val(fdata_dir, s, 'faiss_mrr') for s in SCENARIOS]
        b1 = ax.bar(x - w/2, mrr_model, w, color=colors, alpha=0.9,
                    edgecolor='black', linewidth=0.6, label='TRANP-CNN')
        b2 = ax.bar(x + w/2, mrr_faiss, w, color=colors, alpha=0.3,
                    edgecolor='black', linewidth=0.6, hatch='//', label='FAISS')
        for b, v in zip(b1, mrr_model):
            if v > 0.01:
                ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(SCENARIOS, fontsize=8)
        ax.set_ylabel('MRR', fontsize=9); ax.set_ylim(0, 1.05)
        ax.set_title(f'{src_lbl} → {tgt_lbl}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right'); ax.grid(axis='y', alpha=0.3)

        # ── Row 1: MAP ──
        ax = axes[1, col_idx]
        map_model = [val(data, s, 'MAP') for s in SCENARIOS]
        b3 = ax.bar(x, map_model, 0.5, color=colors, alpha=0.85,
                    edgecolor='black', linewidth=0.6)
        for b, v in zip(b3, map_model):
            if v > 0.01:
                ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(SCENARIOS, fontsize=8)
        ax.set_ylabel('MAP', fontsize=9); ax.set_ylim(0, 1.05)
        ax.set_title(f'{src_lbl} → {tgt_lbl}  (MAP)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # ── Row 2: Top-K recall ──
        ax = axes[2, col_idx]
        top1  = [val(data, s, 'top-1')  for s in SCENARIOS]
        top5  = [val(data, s, 'top-5')  for s in SCENARIOS]
        top10 = [val(data, s, 'top-10') for s in SCENARIOS]
        ax.plot(x, top1,  'o-',  color='#E91E63', label='Top-1',  linewidth=2, markersize=7)
        ax.plot(x, top5,  's--', color='#9C27B0', label='Top-5',  linewidth=2, markersize=7)
        ax.plot(x, top10, '^:',  color='#3F51B5', label='Top-10', linewidth=2, markersize=7)
        ax.set_xticks(x); ax.set_xticklabels(SCENARIOS, fontsize=8)
        ax.set_ylabel('Recall', fontsize=9); ax.set_ylim(0, 1.1)
        ax.set_title(f'{src_lbl} → {tgt_lbl}  (Recall@K)', fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'all_metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ all_metrics_comparison.png")


# ── Plot 2: Rank displacement boxplots ────────────────────────────────────────

def plot_rank_displacement(out_dir, src, tgt):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f'Rank Displacement — {src.split("/")[1]} → {tgt.split("/")[1]}\n'
        f'(positive = model ranked GT file higher than FAISS)',
        fontsize=11, fontweight='bold'
    )
    for i, scenario in enumerate(SCENARIOS):
        ax = axes[i]
        fname = f"{safe_name(src)}_{safe_name(tgt)}_{scenario}_diagnostics.csv"
        fpath = os.path.join(DIAG_DIR, fname)
        if os.path.exists(fpath):
            diag = pd.read_csv(fpath)
            color = SCENARIO_COLORS[scenario]
            ax.boxplot(diag['rank_displacement'], vert=True, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.6),
                       medianprops=dict(color='black', linewidth=2))
            ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel('Rank displacement', fontsize=9)
        ax.set_title(scenario, fontsize=10, fontweight='bold',
                     color=SCENARIO_COLORS[scenario])
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'rank_displacement_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ rank_displacement_boxplot.png")


# ── Plot 3: Commutativity contrast (MRR + MAP side by side) ───────────────────

def plot_commutativity(pair_key, directions, df, out_dir):
    src0, tgt0 = directions[0]
    src1, tgt1 = directions[1]
    d0 = get_pair_rows(df, src0, tgt0)
    d1 = get_pair_rows(df, src1, tgt1)

    x  = np.arange(len(SCENARIOS))
    w  = 0.25
    lbl0 = f'{src0.split("/")[1]}→{tgt0.split("/")[1]}'
    lbl1 = f'{src1.split("/")[1]}→{tgt1.split("/")[1]}'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Commutativity: {src0.split("/")[1]} ↔ {tgt0.split("/")[1]}\n'
        f'Same domain gap, same projects — direction determines performance',
        fontsize=12, fontweight='bold'
    )

    for ax, metric in zip(axes, ['MRR', 'MAP']):
        vals0 = [val(d0, s, metric) for s in SCENARIOS]
        vals1 = [val(d1, s, metric) for s in SCENARIOS]

        b0 = ax.bar(x - w/2, vals0, w, label=lbl0, color='#2196F3', alpha=0.85,
                    edgecolor='black', linewidth=0.7)
        b1 = ax.bar(x + w/2, vals1, w, label=lbl1, color='#E91E63', alpha=0.85,
                    edgecolor='black', linewidth=0.7)

        for b, v in zip(b0, vals0):
            if v > 0.01:
                ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=8)
        for b, v in zip(b1, vals1):
            if v > 0.01:
                ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x); ax.set_xticklabels(SCENARIOS, fontsize=10)
        ax.set_ylabel(metric, fontsize=11); ax.set_ylim(0, 1.0)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'commutativity_contrast.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ commutativity_contrast.png")


# ── Summary CSV ───────────────────────────────────────────────────────────────

def save_pair_summary(directions, df, out_dir):
    rows = []
    for src, tgt in directions:
        subset = df[(df['source_project'] == src) & (df['target_project'] == tgt)].copy()
        rows.append(subset)
    summary = pd.concat(rows).reset_index(drop=True)
    summary.to_csv(os.path.join(out_dir, 'pair_summary.csv'), index=False)
    print(f"  ✓ pair_summary.csv  ({len(summary)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df       = pd.read_csv(RESULTS_CSV)
    # Rename columns to match expected access pattern
    df = df.rename(columns={'model_name': 'model_name'})  # no-op, just confirming load

    # Load FAISS MRR from summary CSV (only column not in phase1 results)
    faiss_df = None
    if os.path.exists(SUMMARY_CSV):
        raw = pd.read_csv(SUMMARY_CSV)
        faiss_df = raw[['source_project', 'target_project', 'scenario', 'faiss_mrr']].copy()
        faiss_df = faiss_df.copy()

    os.makedirs(SHOWCASE_DIR, exist_ok=True)

    for pair_key, directions in SHOWCASE_PAIRS.items():
        out_dir = os.path.join(SHOWCASE_DIR, pair_key)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n── {pair_key} ──")

        # Copy per-bug diagnostic CSVs
        for src, tgt in directions:
            copy_diagnostics(src, tgt, out_dir)

        # Plots
        plot_full_metrics(pair_key, directions, df,
                          faiss_df if faiss_df is not None else pd.DataFrame(),
                          out_dir)
        plot_rank_displacement(out_dir, *directions[0])
        plot_commutativity(pair_key, directions, df, out_dir)
        save_pair_summary(directions, df, out_dir)

    print(f"\n✓ All showcase pairs written to {SHOWCASE_DIR}/")


if __name__ == '__main__':
    main()
