# CPL Phase 1 — Speaker Notes
## Detailed what-to-say for each slide

---

## Slide 1 — Title

**What to say:**
"Today I'm presenting Phase 1 results of my PhD study on Cross-Project Bug Localisation, abbreviated as CPL.
The central question I'm investigating is: can a bug localisation model trained on one project transfer to another? Instead of labelling thousands of bugs in your own project, can you borrow a pre-trained model from a related project and get useful results?

This is practically important because labelling bug report-to-file mappings is expensive and time-consuming. Many real-world projects simply don't have enough historical data to train a reliable bug localiser from scratch.

I'll walk through the setup, the results across five metrics, six key findings, three illustrative showcase pairs, and what this means for source project selection."

---

## Slide 2 — Research Problem & Setup

**What to say:**
"Let me explain the experimental design. I evaluate four scenarios for each source→target project pair.

WP-small represents the baseline: train only on 20% of the target project's bug data. This simulates a project with limited historical information.

WP-large is the upper bound: train on 80% of target data. This simulates a project with abundant labelled data.

CP-cold-start is the hardest case: train entirely on the source project, apply directly to target with zero target data. This is fully zero-shot cross-project transfer.

CP-transfer is the main CPL hypothesis: train on 100% of source data, then fine-tune on 20% of target data — matching the data budget of WP-small exactly.

The core question is: does CP-transfer beat WP-small? If yes, cross-project knowledge helps when target data is limited.

The pipeline: I use BAAI/bge-code-v1 code embeddings to retrieve the top-300 candidate source files via FAISS, then TRANP-CNN re-ranks those 300 candidates using a trained CNN model."

---

## Slide 3 — Metrics Explained

**What to say:**
"Before going to results, I want to be precise about the five metrics I report, because they each capture something different.

Top-K recall asks: is the ground-truth file somewhere in the top K ranked positions? Top-10 = 0.80 means 80% of bugs had the correct file in the top 10. This is the most lenient metric — it gives credit even if the correct file is ranked 9th.

MAP, or Mean Average Precision, is stricter. It measures precision at every rank position where a relevant file appears. A file ranked 1st contributes more to MAP than one ranked 9th. MAP penalises models that retrieve the file but bury it low in the ranking.

MRR, Mean Reciprocal Rank, is the mean of 1 divided by the rank of the first correct answer. MRR = 0.5 means the correct file is at rank 2 on average. MRR = 0.25 means rank 4 on average.

Here's the important implication: Top-K recall and MAP/MRR can give different signals. A model might have high Top-10 recall but low MRR if it consistently puts the correct file at rank 8 or 9. Throughout my analysis I report all five metrics, and I'll point out where they agree and where they diverge."

---

## Slide 4 — Overall Results

**What to say:**
"Here are the mean results across all 92 to 94 source-target pairs.

The pattern is consistent across all five metrics: WP-large is best, then CP-transfer, then WP-small, then CP-cold-start.

The key observation is how small the gap between CP-transfer and WP-large is on Top-K metrics. On Top-10, the gap is only 0.007. On Top-5, it's 0.011. This means CP-transfer retrieves the correct file at nearly the same rate as WP-large.

For MAP and MRR, WP-large has a larger advantage — about 0.02. So WP-large ranks the correct file higher within the retrieved candidates.

CP-cold-start is very low across the board — MAP 0.038, MRR 0.042. Zero-shot transfer without any fine-tuning on the target project has limited practical value in its current form.

The headline finding is that CP-transfer, which uses only 20% of target data — the same budget as WP-small — substantially outperforms WP-small and nearly matches WP-large on recall."

---

## Slide 5 — Finding 1: Wilcoxon Test

**What to say:**
"Let me now go into the statistical significance of the CP-transfer versus WP-small comparison.

First, a note on the statistical test. I use the Wilcoxon signed-rank test because I have paired observations — for each project pair, I have both a CP-transfer result and a WP-small result. This is a non-parametric test, meaning it doesn't assume the data is normally distributed. It works by ranking the differences between paired observations and testing whether positive differences dominate.

I test the one-sided alternative: CP-transfer is greater than WP-small. The null hypothesis is that there's no systematic difference.

The results: for MAP and MRR, I reject the null hypothesis at p < 0.01. CP-transfer is significantly better. The win rates are 64.1% and 67.4% respectively. For Top-5 and Top-10, also significant at p < 0.01.

Critically, Top-1 is NOT significant — p = 0.118. CPL does not help the model put the correct file at rank 1 more often. It helps the model rank it higher in general, which improves MRR and MAP. This is an important nuance.

The interpretation: cross-project pre-training provides a better learned prior for ranking candidate files. The model has seen more diverse bug-file patterns from the source project, which generalises to better relative ordering on the target project."

---

## Slide 6 — Finding 2: CP-transfer vs WP-large

**What to say:**
"Now let me compare CP-transfer against WP-large — the resource-intensive within-project baseline.

The table shows two comparison windows: CP-transfer greater than or equal to WP-large (they're at least tied), and CP-transfer within 10% of WP-large.

For recall metrics: CP-transfer matches or beats WP-large in 52 to 54% of pairs. For Top-10, 63% of pairs are within 10% of WP-large.

For MAP and MRR: WP-large wins in roughly 70% of pairs. The mean gap is 0.023 for MAP and 0.025 for MRR.

What does this mean practically? For use-cases where you're building a developer tool that shows a shortlist of 10 candidate files, CP-transfer is broadly equivalent to WP-large — and it uses 4 times less target training data.

For tasks requiring tight rank-1 precision — say, an automated patch suggestion tool that acts only on the top-ranked file — WP-large retains a meaningful edge.

The practical CPL deployment recommendation is: if your project needs precise top-of-list localisation and you have abundant labelled data, use WP-large. If you're in a data-limited scenario and you need a shortlist, CP-transfer is essentially equivalent."

---

## Slide 7 — Finding 3: Data-Scarce Targets

**What to say:**
"This finding is what I consider the strongest motivating argument for CPL as a practical research contribution.

I split the 94 project pairs by the target project's bug count, using the median of 228 bugs as the threshold. Projects below the median have limited historical data — this is the scenario CPL is designed for.

For these data-scarce targets, the CPL gain — defined as CP-transfer minus WP-small — is 3 to 6 times larger than for data-rich targets. On Top-10 recall, the gain is +0.043 for few-bug targets versus +0.007 for many-bug targets. On MRR, +0.032 versus +0.012.

The implication is direct: if you have a project with fewer than 200 historical bugs, using CPL is significantly better than training from scratch on that limited data. The benefit diminishes as the target project accumulates more bug data, which makes intuitive sense — at some point, within-project data is sufficient.

This motivates CPL as the recommended default strategy for new projects, projects that have recently changed technology stacks, or niche projects with sparse bug histories."

---

## Slide 8 — Finding 4: Spearman Correlations

**What to say:**
"I used Spearman rank correlation to understand which features of the source and target projects predict CPL performance. Let me first explain Spearman correlation for context.

Spearman's ρ measures the monotonic relationship between two variables. Unlike Pearson, it works on ranks rather than raw values, making it robust to outliers and non-linear relationships. ρ = +1 means as one variable increases, the other always increases. ρ = −1 means they're inversely related. ρ = 0 means no systematic relationship.

In our case, each observation is a source→target pair. I'm asking: do project-level features predict how well CP-transfer performs on that pair?

The standout finding is tgt_LoC, the target codebase size. ρ = −0.855 for Top-10 recall. This is an extremely strong negative correlation — as the target codebase gets larger, Top-10 recall drops very consistently across all 94 pairs. For MAP and MRR, ρ is −0.65 and −0.67, also very strong.

Bug report verbosity is the second-strongest feature at ρ = +0.383 for MAP — more detailed bug reports help the model match bug descriptions to source files.

Critically, source-side features are weak or non-significant. Source bug count has ρ = 0.07 for MRR. Domain gap — the logistic regression accuracy I use as a dissimilarity measure between projects — has ρ = 0.038 for MRR, which is not significant.

The practical conclusion: when deciding whether to deploy CPL on a project, look at the target codebase size and bug report quality. Source selection matters much less than these target properties."

---

## Slide 9 — Finding 5: FAISS Ceiling

**What to say:**
"This is a diagnostic finding that shapes the entire research agenda going forward.

The pipeline works in two stages: FAISS retrieves the top-300 candidate files using embedding similarity, then TRANP-CNN reranks those 300 candidates.

If the ground-truth file is NOT in the top-300 retrieved by FAISS, the model cannot possibly find it — it can only rerank what it's given. This is what I call an 'unreachable' case.

The finding is that pct_unreachable equals exactly 0% across all 94 pairs. Every single ground-truth file was retrieved by FAISS in its top-300 candidates.

This is extremely informative because it means every single failure in my results — every case where the correct file was not in the model's top-10 — is a model ranking failure, not a retrieval failure. FAISS with BAAI/bge-code-v1 embeddings is already good enough to find the correct file; the challenge is ranking it high enough.

The implication for future work is clear: improving the reranking model architecture — replacing the CNN with a transformer, using contrastive learning, or using a GNN — will directly improve all metrics. There's no point in increasing the retrieval pool beyond 300. The FAISS embeddings are not the bottleneck."

---

## Slide 10 — Finding 6: Commutativity (Pair A)

**What to say:**
"This showcase pair illustrates one of the most counterintuitive findings: CPL is not commutative. The same two projects in opposite directions can give very different results.

Matplotlib-to-jupyterlab: CP-transfer achieves MRR 0.614, which is actually the best scenario — even better than WP-small's 0.438 and WP-large's 0.554.

Jupyterlab-to-matplotlib: CP-transfer drops to MRR 0.236, actually worse than WP-small at 0.372.

What explains this? Both pairs have identical domain gap — 0.972 — and they use the exact same two projects, just swapped.

The explanation is entirely target codebase size. When jupyterlab is the target — 39,000 lines of code — it's a compact, well-structured codebase where the correct file can rise to the top-10 easily. When matplotlib is the target — 249,000 lines of code — the correct file is competing against far more candidates and is much harder to rank in the top-10.

This finding also shows that WP-large can be worse than WP-small in some cases. For jupyterlab-to-matplotlib, WP-large gives MRR 0.294 while WP-small gives 0.372. This suggests overfitting: with more training data on matplotlib's specific bug patterns, the model loses generalisability on the test bugs.

The general rule: the target project's codebase size determines the difficulty of localisation, not the source. Commutativity breaks along the tgt_LoC axis."

---

## Slide 11 — Showcase Pair B: Best Performer

**What to say:**
"Numpy-to-jupyterlab is our best-performing pair and provides the strongest evidence for CPL feasibility.

WP-large achieves MRR 0.724 and Top-5 recall of 100% — every single jupyterlab bug is found in the top-5. CP-transfer reaches MRR 0.666 with the same 100% Top-5 recall.

The gap between CP-transfer at 0.666 and WP-large at 0.724 is only 0.058, or about 8%. This is achieved with the same 20% target data budget as WP-small, which only gets MRR 0.464.

The FAISS baseline for this pair is MRR 0.022. The model achieves 33 times that. This is not a subtle improvement — the model is doing substantial work to reorganise the ranking from near-random to highly accurate.

The zero-shot baseline, CP-cold-start, reaches MRR 0.258 using only numpy training data and zero jupyterlab examples. That's non-trivial — over 25% reciprocal rank purely from cross-project transfer before any fine-tuning.

Why does this pair work so well? Jupyterlab is a 39,000-line Python project with 197 well-documented bug reports. Small, focused codebase plus verbose bug reports plus a rich source project like numpy — this is the ideal CPL scenario."

---

## Slide 12 — Showcase Pair C: Worst Performer

**What to say:**
"Numpy-to-scipy is the opposite extreme. Every scenario gets zero Top-1, Top-5, and Top-10 recall. MRR peaks at 0.029 for WP-large.

Before concluding this is a model failure, let me diagnose it carefully.

Scipy has 438,000 lines of code — the largest codebase in our dataset. pct_unreachable is 0% — the correct file is in FAISS's top-300. The problem is not retrieval.

Looking at rank displacement: the model IS improving the ranking. WP-large has a mean rank displacement of +131 — the model pushes the correct file up 131 positions on average. But the file starts at around rank 167 — so after improvement it lands around rank 36, which is still outside top-10.

What would it need? The model would need to displace the correct file by 157 positions to break into top-10. At 131 positions average improvement, it's close but not there yet.

The key point is this: no source project selection will fix this. I tested 11 different source projects for scipy, and the oracle MRR — the best possible source — is still only 0.024. This is a target viability failure. The codebase is simply too large for the current architecture.

This motivates the architectural future work: hierarchical rerankers, better localisation at scale, or retrieval augmentation specifically for large codebases."

---

## Slide 13 — Source Selection (Kendall τ)

**What to say:**
"Given that we have multiple potential source projects, can we predict which one will produce the best CP-transfer results?

I evaluate this using Kendall's tau rank correlation. For each target project with at least 3 source candidates, I rank the sources by a heuristic score and compare that ranking to the actual oracle ranking — the ordering by achieved CP-transfer MRR.

Kendall's tau ranges from -1 to +1. τ = 1 means the heuristic perfectly identifies the best source. τ = 0 means the heuristic is random. Negative τ means the heuristic is worse than random.

The most-bugs heuristic — simply picking the source with the most bug reports — achieves Hit@1 of 41.7% and Kendall τ of +0.253 across 12 targets.

This is better than I initially expected. The heuristic correctly identifies the best source in 5 out of 12 cases and has a positive ordering signal — larger sources genuinely tend to produce better transfer results more often. It captures about 53% of the gap between random and oracle selection.

However, Hit@1 of 41.7% means the heuristic fails 58% of the time. This is a partially solved problem, not a fully solved one. Source selection requires a learned approach — a meta-model trained on historical pair outcomes — to close the remaining gap.

The two negative results I want to highlight: domain gap is NOT a useful filter for source selection, and bug report similarity is NOT a useful proxy. Neither shows significant correlation with achieved performance. These intuitive proxies fail empirically."

---

## Slide 14 — Decision Framework

**What to say:**
"Based on all the findings, I propose a two-phase decision framework for practitioners deploying CPL.

Phase 1 is target viability. Before selecting a source project, check whether the target is a suitable CPL candidate at all.

If the target codebase exceeds 200,000 lines of code — flag it as high-risk. Our Spearman correlation ρ = −0.855 shows this is the dominant factor. If the target is very large, even the best source will produce mediocre results.

If bug reports are very terse — less than 30 words on average — the model has insufficient textual signal to match bug descriptions to files.

If the target already has over 300 historical bugs, WP-large may actually be more reliable than CP-transfer, since the CPL gain shrinks for data-rich targets.

Phase 2 is source selection. If the target passes the viability check, apply these heuristics in order:

First, default to the source with the most bug reports. This achieves 41.7% Hit@1 and is the best simple heuristic available.

Second, among similar-sized sources, prefer larger codebases — src_LoC has a weak positive correlation.

Third, ignore domain gap — it has no significant predictive power for any metric.

Fourth, ignore bug similarity metrics — they showed no reliable correlation in our dataset.

I want to be clear that this is a set of heuristics, not an algorithm. The most-bugs heuristic fails 58% of the time. Learned source selection using a meta-model is an open research problem."

---

## Slide 15 — Contributions & Implications

**What to say:**
"Let me summarise the research contributions and what they mean for the field.

First: CPL significantly improves ranking quality over WP-small with p = 0.005. This is the primary feasibility claim. The literature already established that CPL is possible; we now have a quantified, statistically validated improvement with matched data budgets.

Second: CP-transfer achieves WP-large-level recall in over half of pairs. This is relevant for practitioners who care about building shortlists rather than exact rank-1 localisation.

Third: the gain is 3 to 4 times larger for data-scarce targets. This positions CPL as the recommended strategy for new projects, not as a general-purpose improvement over well-resourced within-project training.

Fourth: target LoC is the dominant predictor at ρ = −0.855. This means that when we evaluate CPL, we must stratify by target codebase size. A study that averages across scipy and jupyterlab will understate the effectiveness of CPL for small targets and overstate it for large ones.

Fifth: the most-bugs heuristic achieves Hit@1 = 41.7% for source selection. This establishes an empirical baseline for future learned source selection methods to beat.

Sixth: pct_unreachable = 0% frames the research agenda. Future work should focus on the reranking model, not the retrieval step."

---

## Slide 16 — Limitations & Future Work

**What to say:**
"I want to be transparent about the limitations.

The mean MRR values of 0.20 to 0.25 look low compared to within-project SOTA papers that report MRR above 0.5. But this comparison is unfair — those papers train and test on the same project, which is fundamentally easier. Our correct comparison is our own FAISS baseline and WP-small, not cross-study numbers.

The results cover Python and Java projects. I expect the qualitative findings to hold across languages — code embeddings are language-aware — but this hasn't been empirically validated.

Large-codebase targets like scipy dominate the low-performance tail and pull down averages. Stratified reporting by target size gives a more honest picture.

For future work, the highest priority is a learned source selection meta-model. Our heuristic captures 53% of the oracle gap — a learned approach could close that further.

Multi-source ensembling is another natural direction: instead of picking the single best source, combine signal from multiple sources. This is analogous to ensemble methods in other transfer learning settings.

The most impactful architectural improvement would be replacing TRANP-CNN's CNN reranker with a transformer-based model. Given that all GT files are retrievable and all failures are ranking failures, a stronger reranker will directly lift all metrics."

---

## Slide 17 — Summary Takeaways

**What to say:**
"To summarise the six key takeaways:

One: CP-transfer beats WP-small in 67.4% of pairs on MRR with p = 0.005. The improvement is statistically significant and consistent.

Two: CP-transfer matches WP-large on Top-10 recall in 53% of pairs. For shortlist-based applications, CP-transfer is essentially equivalent to WP-large with 4 times less labelled data.

Three: CPL is 3 to 4 times more valuable for data-scarce targets — projects with fewer than 228 bugs see substantially larger gains.

Four: target codebase size is the single strongest predictor of achievable performance, with ρ = −0.855 for Top-10. Check the target before deploying CPL.

Five: source selection is partially solved. The most-bugs heuristic achieves 41.7% Hit@1 and positive Kendall τ of +0.25. Domain gap and similarity metrics are not useful for source selection.

Six: the FAISS ceiling is not the bottleneck — pct_unreachable = 0%. The remaining challenge is the reranking architecture, and improving it will directly translate to better results.

Thank you — I'm happy to take questions."

---

## Common PhD Viva / Presentation Questions

**Q: Why is CP-cold-start so low?**
A: Zero-shot cross-project transfer means the model has never seen the target project's code structure, naming conventions, or bug patterns. Fine-tuning on even 20% of target data dramatically changes the loss landscape. CP-cold-start serves as a floor — it shows that some cross-project signal exists, but fine-tuning is necessary to unlock it.

**Q: Why not use more FAISS candidates — say, top-500?**
A: pct_unreachable = 0% shows that top-300 already captures all GT files. Increasing to 500 would not improve recall — the model would just have 200 more non-relevant candidates to rank around, which could dilute the reranking signal.

**Q: Couldn't the difference between CP-transfer and WP-small just be random noise?**
A: That's exactly what the Wilcoxon signed-rank test addresses. With 92 paired observations and p = 0.005 for MRR, the probability of observing this pattern by chance is 0.5%. The test is conservative and non-parametric, so it doesn't assume any particular data distribution.

**Q: Why did you choose Spearman correlation over Pearson?**
A: Pearson assumes a linear relationship and is sensitive to outliers. Our data has projects spanning a very wide range of sizes — from 2,000 to 438,000 LoC. Spearman operates on ranks, which is more robust in this heterogeneous setting. It asks 'is there a monotonic relationship?' rather than 'is there a linear relationship?', which is the more appropriate question here.

**Q: Domain gap of 0.97 — isn't that extreme? Doesn't that mean projects are completely different?**
A: A domain gap of 0.97 means a logistic regression classifier can distinguish embeddings from the two projects with 97% accuracy — yes, they are structurally very different. But the key finding is that domain gap does not predict CPL performance (ρ < 0.12, non-significant). Dissimilar projects can still transfer well because TRANP-CNN learns structural patterns — the relationship between bug description tokens and source file tokens — which are universal across programming contexts.

**Q: What's the difference between MAP and MRR in practice?**
A: MRR gives full credit for the first correct answer only. MAP gives partial credit at every rank position where a relevant file appears. For bug localisation, bugs typically have one ground-truth file, so MAP and MRR behave similarly — but MAP is slightly more standard in information retrieval literature and penalises models that put the correct file just above the threshold more harshly.
