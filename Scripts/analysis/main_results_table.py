"""
main_results_table.py
──────────────────────
Generates the primary results tables for the JSS paper.

Table 1 (scenario_means): Mean metrics per model × scenario across all paper pairs.
    → This is Table 3 in the paper: the headline performance comparison.

Table 2 (wilcoxon_tests): Pairwise Wilcoxon signed-rank tests for the key
    CPL-vs-WP comparisons, per model and per metric.
    → Provides the statistical significance evidence for RQ1.

Table 3 (cpl_gain_summary): Per-model CPL gain statistics (win rate, median
    gain among winners, median loss among losers).

All tables are saved as CSV and printed in LaTeX-ready format.

Output
──────
results/main_results_scenario_means.csv
results/main_results_wilcoxon_tests.csv
results/main_results_cpl_gain_summary.csv

Run
───
    python Scripts/analysis/main_results_table.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_CSV = "/home/cs21d002_eashaan/PhD/Objective1/results/paper_results_complete.csv"
OUT_DIR     = "/home/cs21d002_eashaan/PhD/Objective1/results"

SCENARIOS       = ['WP-small', 'WP-large', 'CP-cold-start', 'CP-transfer']
METRICS         = ['MRR', 'MAP', 'top-1', 'top-5', 'top-10']
METRIC_DISPLAY  = {'MRR': 'MRR', 'MAP': 'MAP', 'top-1': 'Top-1',
                   'top-5': 'Top-5', 'top-10': 'Top-10'}

# Key comparisons for RQ1 significance tests
COMPARISONS = [
    ('CP-transfer',   'WP-small',    'CPT vs WPS'),
    ('CP-transfer',   'WP-large',    'CPT vs WPL'),
    ('CP-cold-start', 'WP-small',    'CPC vs WPS'),
    ('WP-large',      'WP-small',    'WPL vs WPS'),
    ('CP-transfer',   'CP-cold-start','CPT vs CPC'),
]


def load_data():
    return pd.read_csv(RESULTS_CSV)


# ── Table 1: Scenario means ───────────────────────────────────────────────────

def build_scenario_means(df):
    rows = []
    for model, mg in df.groupby('model_name'):
        for scenario in SCENARIOS:
            sg = mg[mg['scenario'] == scenario]
            row = {'model': model, 'scenario': scenario, 'n_pairs': len(sg)}
            for m in METRICS:
                row[f'mean_{m}']   = sg[m].mean()   if len(sg) > 0 else np.nan
                row[f'median_{m}'] = sg[m].median() if len(sg) > 0 else np.nan
                row[f'std_{m}']    = sg[m].std()    if len(sg) > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def print_scenario_means(means_df):
    print("\n" + "="*70)
    print("TABLE 1 — Mean Performance per Model × Scenario")
    print("="*70)
    for model, mg in means_df.groupby('model'):
        print(f"\n  {model}")
        print(f"  {'Scenario':<18} " + "  ".join(f"{METRIC_DISPLAY[m]:>8}" for m in METRICS))
        print(f"  {'-'*18} " + "  ".join('-'*8 for _ in METRICS))
        scenario_order = {s: i for i, s in enumerate(SCENARIOS)}
        mg_sorted = mg.sort_values('scenario', key=lambda x: x.map(scenario_order))
        for _, row in mg_sorted.iterrows():
            vals = "  ".join(f"{row[f'mean_{m}']:8.4f}" for m in METRICS)
            print(f"  {row['scenario']:<18} {vals}  (n={int(row['n_pairs'])})")


def latex_scenario_means(means_df):
    """Print a LaTeX-ready table (copy into paper as-is)."""
    print("\n" + "─"*70)
    print("LaTeX Table 1 (paste into paper):")
    print("─"*70)
    header_cols = " & ".join(METRIC_DISPLAY[m] for m in METRICS)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Mean bug localization performance per scenario and model.}")
    print(r"\label{tab:main-results}")
    print(r"\begin{tabular}{ll" + "r" * len(METRICS) + "}")
    print(r"\toprule")
    print(f"Model & Scenario & {header_cols} \\\\")
    print(r"\midrule")
    for model, mg in means_df.groupby('model'):
        scenario_order = {s: i for i, s in enumerate(SCENARIOS)}
        mg_sorted = mg.sort_values('scenario', key=lambda x: x.map(scenario_order))
        first = True
        for _, row in mg_sorted.iterrows():
            model_cell = f"\\multirow{{{len(mg)}}}{{*}}{{{model}}}" if first else ""
            vals = " & ".join(f"{row[f'mean_{m}']:.4f}" for m in METRICS)
            print(f"{model_cell} & {row['scenario']} & {vals} \\\\")
            first = False
        print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ── Table 2: Wilcoxon significance tests ──────────────────────────────────────

def build_wilcoxon_tests(df):
    rows = []
    for model, mg in df.groupby('model_name'):
        for sc_a, sc_b, label in COMPARISONS:
            for metric in METRICS:
                a = (mg[mg['scenario'] == sc_a]
                     .set_index(['source_project', 'target_project'])[metric])
                b = (mg[mg['scenario'] == sc_b]
                     .set_index(['source_project', 'target_project'])[metric])
                common = a.index.intersection(b.index)
                if len(common) < 5:
                    continue
                diffs = (a[common] - b[common]).dropna().values
                if len(diffs) < 5:
                    continue

                n_pos = int((diffs > 0).sum())
                n_neg = int((diffs < 0).sum())
                n_tie = int((diffs == 0).sum())

                try:
                    _, p_gt = wilcoxon(diffs, alternative='greater', zero_method='zsplit')
                except ValueError:
                    p_gt = np.nan
                try:
                    _, p_two = wilcoxon(diffs, alternative='two-sided', zero_method='zsplit')
                except ValueError:
                    p_two = np.nan

                sig = ('***' if (pd.notna(p_gt) and p_gt < 0.001) else
                       ('**'  if (pd.notna(p_gt) and p_gt < 0.01)  else
                        ('*'   if (pd.notna(p_gt) and p_gt < 0.05)  else 'ns')))

                rows.append({
                    'model':         model,
                    'comparison':    label,
                    'metric':        metric,
                    'n_pairs':       len(common),
                    'n_wins':        n_pos,
                    'n_ties':        n_tie,
                    'n_losses':      n_neg,
                    'win_rate':      n_pos / len(common),
                    'mean_delta':    float(np.mean(diffs)),
                    'median_delta':  float(np.median(diffs)),
                    'p_greater':     p_gt,
                    'p_two_sided':   p_two,
                    'significance':  sig,
                })
    return pd.DataFrame(rows)


def print_wilcoxon_tests(wilcoxon_df):
    print("\n" + "="*70)
    print("TABLE 2 — Wilcoxon Signed-Rank Tests (H1: sc_a > sc_b)")
    print("="*70)
    key_metric = 'MRR'
    sub = wilcoxon_df[wilcoxon_df['metric'] == key_metric].copy()
    print(f"\n  (Showing MRR only — see CSV for all metrics)")
    print(f"  {'Model':<12} {'Comparison':<15} {'n':>4} {'Win%':>6} "
          f"{'Mean Δ':>8} {'Median Δ':>9} {'p (>)':>8} {'Sig':>5}")
    print(f"  {'-'*12} {'-'*15} {'-'*4} {'-'*6} {'-'*8} {'-'*9} {'-'*8} {'-'*5}")
    for _, row in sub.iterrows():
        p_str = f"{row['p_greater']:.4f}" if pd.notna(row['p_greater']) else "  N/A"
        print(f"  {row['model']:<12} {row['comparison']:<15} {row['n_pairs']:>4} "
              f"{row['win_rate']:>6.1%} {row['mean_delta']:>8.4f} "
              f"{row['median_delta']:>9.4f} {p_str:>8} {row['significance']:>5}")


def latex_wilcoxon_tests(wilcoxon_df):
    print("\n" + "─"*70)
    print("LaTeX Table 2 (Wilcoxon, MRR only — extend for other metrics):")
    print("─"*70)
    sub = wilcoxon_df[wilcoxon_df['metric'] == 'MRR']
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Wilcoxon signed-rank tests: $H_1$: scenario A $>$ scenario B (MRR).}")
    print(r"\label{tab:wilcoxon}")
    print(r"\begin{tabular}{llrrrrl}")
    print(r"\toprule")
    print(r"Model & Comparison & $n$ & Win\% & Mean $\Delta$ & Median $\Delta$ & $p$ \\")
    print(r"\midrule")
    for _, row in sub.iterrows():
        p_str = f"{row['p_greater']:.4f}" if pd.notna(row['p_greater']) else "--"
        sig   = row['significance']
        print(f"{row['model']} & {row['comparison']} & {row['n_pairs']} & "
              f"{row['win_rate']:.1%} & {row['mean_delta']:.4f} & "
              f"{row['median_delta']:.4f} & {p_str}{sig} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ── Table 3: CPL gain magnitude summary ───────────────────────────────────────

def build_cpl_gain_summary(df):
    rows = []
    for model, mg in df.groupby('model_name'):
        cpt = (mg[mg['scenario'] == 'CP-transfer']
               .set_index(['source_project', 'target_project']))
        wps = (mg[mg['scenario'] == 'WP-small']
               .set_index(['source_project', 'target_project']))
        common = cpt.index.intersection(wps.index)
        if len(common) == 0:
            continue
        for metric in METRICS:
            diffs = (cpt.loc[common, metric] - wps.loc[common, metric]).dropna().values
            n_pos = (diffs > 0).sum()
            n_neg = (diffs < 0).sum()
            rows.append({
                'model':            model,
                'metric':           metric,
                'n_pairs':          len(diffs),
                'win_rate':         n_pos / len(diffs) if len(diffs) > 0 else np.nan,
                'mean_delta':       float(np.mean(diffs)),
                'median_delta':     float(np.median(diffs)),
                'median_win':       float(np.median(diffs[diffs > 0])) if n_pos > 0 else np.nan,
                'median_loss':      float(np.median(diffs[diffs < 0])) if n_neg > 0 else np.nan,
                'n_wins':           int(n_pos),
                'n_losses':         int(n_neg),
                'asymmetry_ratio':  (abs(np.median(diffs[diffs > 0])) /
                                     abs(np.median(diffs[diffs < 0])))
                                     if n_pos > 0 and n_neg > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def print_cpl_gain_summary(gain_df):
    print("\n" + "="*70)
    print("TABLE 3 — CPL Gain Magnitude (CP-transfer − WP-small)")
    print("="*70)
    print(f"  {'Model':<10} {'Metric':<8} {'WinRate':>8} {'MeanΔ':>8} "
          f"{'MedWin':>8} {'MedLoss':>9} {'AsymRatio':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")
    for _, row in gain_df.iterrows():
        med_l = f"{row['median_loss']:.4f}" if pd.notna(row['median_loss']) else "   N/A"
        asym  = f"{row['asymmetry_ratio']:.2f}×" if pd.notna(row['asymmetry_ratio']) else "   N/A"
        print(f"  {row['model']:<10} {row['metric']:<8} {row['win_rate']:>8.1%} "
              f"{row['mean_delta']:>8.4f} {row['median_win']:>8.4f} "
              f"{med_l:>9} {asym:>10}")
    print("\n  AsymRatio > 1 means median wins are larger than median losses "
          "(desirable for CPL adoption).")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_data()
    models_found = df['model_name'].unique().tolist()
    print(f"  Models in data: {models_found}")
    print(f"  Total rows: {len(df)}")

    print("\nBuilding Table 1: Scenario means...")
    means_df = build_scenario_means(df)
    print_scenario_means(means_df)
    latex_scenario_means(means_df)
    out = os.path.join(OUT_DIR, 'main_results_scenario_means.csv')
    means_df.to_csv(out, index=False)
    print(f"\n  ✓ {out}")

    print("\nBuilding Table 2: Wilcoxon significance tests...")
    wilcoxon_df = build_wilcoxon_tests(df)
    print_wilcoxon_tests(wilcoxon_df)
    latex_wilcoxon_tests(wilcoxon_df)
    out = os.path.join(OUT_DIR, 'main_results_wilcoxon_tests.csv')
    wilcoxon_df.to_csv(out, index=False)
    print(f"\n  ✓ {out}")

    print("\nBuilding Table 3: CPL gain magnitude...")
    gain_df = build_cpl_gain_summary(df)
    print_cpl_gain_summary(gain_df)
    out = os.path.join(OUT_DIR, 'main_results_cpl_gain_summary.csv')
    gain_df.to_csv(out, index=False)
    print(f"\n  ✓ {out}")

    print("\n✓ All tables done.")


if __name__ == '__main__':
    main()
