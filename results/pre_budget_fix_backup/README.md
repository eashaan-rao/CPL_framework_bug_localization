# Pre-fix baseline of the primary results files

These are the three primary results files exactly as of commit `233ddd4`, i.e. **before any WP-small
budget correction** — the WP-small rows here are the original, buggy 10%-of-target-bugs runs.

This is the correct baseline for any before/after comparison.

**Do not** use commit `f9a0a44` as a baseline: that intermediate state had already merged
**rep1-only** 20%-budget WP-small values, so a diff against it measures "rep1 vs final estimate",
not "10% budget vs 20% budget".

Verified at restore time: the non-WP-small rows (567 / 747 / 747) are byte-identical between this
baseline and the corrected files, so the only thing the correction changes is the WP-small rows.

Regenerate with:

    git show 233ddd4:results/<file>.csv
