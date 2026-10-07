"""
mixed_effects_check.py
Linear mixed-effects model of CPL gain (CP-transfer MRR - WP-small MRR) on model identity,
target project as random intercept. Supports the Conclusion-validity paragraph of the paper.
"""
import pandas as pd
import statsmodels.formula.api as smf

ROOT = "/home/cs21d002_eashaan/PhD/Objective1"
df = pd.read_csv(f"{ROOT}/results/negative_transfer_analysis.csv")
df["model"] = pd.Categorical(df["model_name"], ["BLAZE", "COOBA", "TRANP-CNN"])
res = smf.mixedlm("delta_mrr ~ C(model)", df, groups=df["target_project"]).fit(reml=True)
print(res.summary())
bv = float(res.cov_re.iloc[0, 0]); rv = float(res.scale)
print(f"between-project variance={bv:.4f} residual={rv:.4f} ICC={bv/(bv+rv):.3f}")

# Per-model random-intercept test of mean CPL gain (cluster-aware analogue of the RQ1 test)
for m, g in df.groupby("model_name"):
    r = smf.mixedlm("delta_mrr ~ 1", g, groups=g["target_project"]).fit(reml=True)
    b = float(r.cov_re.iloc[0, 0])
    print(f"{m}: intercept={r.params['Intercept']:.4f} se={r.bse['Intercept']:.4f} p={r.pvalues['Intercept']:.4f} ICC={b/(b+r.scale):.2f}")

# Target-level aggregation (n = 13 targets): one-sided Wilcoxon on per-target mean gain
from scipy.stats import wilcoxon
for m, g in df.groupby("model_name"):
    t = g.groupby("target_project")["delta_mrr"].mean()
    print(f"{m}: target-level mean gain={t.mean():.4f}, positive targets={(t>0).sum()}/13, "
          f"Wilcoxon p={wilcoxon(t, alternative='greater').pvalue:.4f}")
