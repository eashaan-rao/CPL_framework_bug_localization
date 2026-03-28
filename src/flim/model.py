"""
FLIM Adaptive Learning-to-Rank (LTR) model.

Faithfully adapts the Adaptive_Process from the original FLIM codebase:
  1. 2-fold CV selects the best feature-weighting method.
  2. 2-fold CV selects the best sklearn regression model.
  3. Final model is trained on all available training data.

Columns are configurable so this ranker works with any feature set
(original FLIM uses 4 semantic + 19 IR; our adaptation uses 4 semantic).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor, Ridge, ElasticNet
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import KFold

# Default feature columns (overridden by pipeline as needed)
DEFAULT_FEATURE_COLUMNS = ['f1', 'f2', 'f3', 'f4']


# ─────────────────────────────────────────────────────────────────────────────
#  Metric helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_map(df: pd.DataFrame, scores: np.ndarray) -> float:
    """Compute MAP given a DataFrame indexed by (bug_id, …) and predicted scores."""
    tmp = df.copy()
    tmp['_score'] = scores
    ap_list = []
    for _, group in tmp.groupby(level=0, sort=False):
        ranked = group.sort_values('_score', ascending=False)
        labels = ranked['used_in_fix'].values
        precisions, n_rel = [], 0
        for i, lbl in enumerate(labels):
            if lbl == 1:
                n_rel += 1
                precisions.append(n_rel / (i + 1))
        ap_list.append(float(np.mean(precisions)) if precisions else 0.0)
    return float(np.mean(ap_list)) if ap_list else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Feature-weighting methods  (same set as original Adaptive_Process)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(w: np.ndarray) -> np.ndarray:
    s = w.sum()
    return w / s if s > 0 else w


def _w_extra_trees(df, cols):
    clf = ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(df[cols], df['used_in_fix'])
    return _normalise(clf.feature_importances_)


def _w_gradient_boost(df, cols):
    reg = GradientBoostingRegressor(n_estimators=50, random_state=42)
    reg.fit(df[cols], df['used_in_fix'])
    return _normalise(reg.feature_importances_)


def _w_chi2(df, cols):
    X = df[cols].clip(lower=0)
    scores, _ = chi2(X, df['used_in_fix'])
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0)
    return _normalise(scores)


def _w_mutual_info(df, cols):
    scores = mutual_info_classif(
        df[cols], df['used_in_fix'], discrete_features=False, random_state=42
    )
    return _normalise(scores)


def _w_constant(df, cols):
    return np.ones(len(cols)) / len(cols)


WEIGHT_METHODS = {
    'extra_trees':    _w_extra_trees,
    'gradient_boost': _w_gradient_boost,
    'chi2':           _w_chi2,
    'mutual_info':    _w_mutual_info,
    'constant':       _w_constant,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Regression model candidates
# ─────────────────────────────────────────────────────────────────────────────

def _get_regressors():
    return [
        Ridge(alpha=1.0),
        Ridge(alpha=0.1),
        ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
        SGDRegressor(loss='epsilon_insensitive', penalty='elasticnet',
                     alpha=1e-4, max_iter=1000, random_state=42),
        SGDRegressor(loss='huber', penalty='l2',
                     alpha=1e-4, max_iter=1000, random_state=42),
        SGDRegressor(loss='squared_error', penalty='l2',
                     alpha=1e-4, max_iter=1000, random_state=42),
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  FLIMRanker
# ─────────────────────────────────────────────────────────────────────────────

class FLIMRanker:
    """
    Adaptive Learning-to-Rank for FLIM.

    Training (fit):
        Step 1 — 2-fold CV over 5 feature-weighting methods → best weights
        Step 2 — 2-fold CV over 6 regression models         → best regressor
        Step 3 — Fit final model on all training data

    Inference (predict_scores):
        Returns a relevance score per row; files ranked per bug by
        descending score.

    Parameters
    ----------
    columns : list[str]
        Feature column names to use.  Defaults to DEFAULT_FEATURE_COLUMNS.
    cv_folds : int
        Number of CV folds for hyperparameter selection (default 2).
    """

    def __init__(
        self,
        columns:  list[str] | None = None,
        cv_folds: int = 2,
    ):
        self.columns  = columns if columns is not None else DEFAULT_FEATURE_COLUMNS
        self.cv_folds = cv_folds

        self.weight_method_name: str          = 'constant'
        self.weights:            np.ndarray | None = None
        self.regressor                             = None
        self.use_regressor:      bool         = False

        self._weight_score: float = 0.0
        self._reg_score:    float = 0.0

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _cv_score_weights(self, df: pd.DataFrame, method) -> float:
        """Return mean MAP for a feature-weighting method via k-fold CV."""
        kf = KFold(n_splits=self.cv_folds, shuffle=False)
        scores = []
        for tr_idx, va_idx in kf.split(df):
            tr, va = df.iloc[tr_idx], df.iloc[va_idx]
            try:
                w      = method(tr, self.columns)
                base   = np.dot(tr[self.columns].values, w)
                target = base + tr['used_in_fix'].values * np.abs(base).max()
                reg    = Ridge(alpha=1.0)
                reg.fit(tr[self.columns], target)
                scores.append(_compute_map(va, reg.predict(va[self.columns])))
            except Exception:
                scores.append(0.0)
        return float(np.mean(scores))

    def _cv_score_regressor(
        self, df: pd.DataFrame, reg, target: np.ndarray
    ) -> float:
        """Return mean MAP for a regression model via k-fold CV."""
        kf = KFold(n_splits=self.cv_folds, shuffle=False)
        scores = []
        for tr_idx, va_idx in kf.split(df):
            try:
                reg.fit(df.iloc[tr_idx][self.columns], target[tr_idx])
                scores.append(
                    _compute_map(df.iloc[va_idx],
                                 reg.predict(df.iloc[va_idx][self.columns]))
                )
            except Exception:
                scores.append(0.0)
        return float(np.mean(scores))

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> None:
        """
        Train the FLIM ranker.

        Parameters
        ----------
        df : DataFrame indexed by (bug_id, blob_sha)
             Must contain columns self.columns and 'used_in_fix' (1=buggy, 0=not).
        """
        if len(df) < 4:
            self.weights      = _w_constant(df, self.columns)
            self.use_regressor = False
            return

        # ── Step 1: best feature-weighting method ────────────────────────
        best_w_name  = 'constant'
        best_w_score = -1.0
        best_w       = _w_constant(df, self.columns)

        for name, method in WEIGHT_METHODS.items():
            try:
                score = self._cv_score_weights(df, method)
                if score > best_w_score:
                    best_w_score = score
                    best_w_name  = name
                    best_w       = method(df, self.columns)   # refit on full data
            except Exception:
                continue

        self.weight_method_name = best_w_name
        self.weights            = best_w
        self._weight_score      = best_w_score

        # ── Step 2: best regressor ────────────────────────────────────────
        base_scores = np.dot(df[self.columns].values, self.weights)
        target      = base_scores + df['used_in_fix'].values * np.abs(base_scores).max()

        best_reg       = None
        best_reg_score = -1.0

        for reg in _get_regressors():
            try:
                score = self._cv_score_regressor(df, reg, target)
                if score > best_reg_score:
                    best_reg_score = score
                    best_reg       = reg
            except Exception:
                continue

        self._reg_score    = best_reg_score
        self.use_regressor = (best_reg is not None) and (best_reg_score >= best_w_score)

        if self.use_regressor and best_reg is not None:
            self.regressor = best_reg
            self.regressor.fit(df[self.columns], target)

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict relevance scores for (bug_id, blob_sha) pairs.

        Returns
        -------
        np.ndarray of float scores, one per row (higher = more relevant).
        """
        if self.weights is None:
            return np.zeros(len(df))

        if self.use_regressor and self.regressor is not None:
            try:
                return self.regressor.predict(df[self.columns])
            except Exception:
                pass

        return np.dot(df[self.columns].values, self.weights)
