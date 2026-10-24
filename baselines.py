"""Baseline imputers used in Table 5.2 of the thesis (MissForest, MICE,
KNNImputer, IterativeImputer). The thesis does not publish the code or settings
used for these baselines, so library defaults are used (see README).

All baselines work on the same normalised/label-encoded frame that TGAIN uses
and are fit on the whole frame (complete training rows + incomplete test rows).
Categorical codes are rounded to the nearest valid code afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

from .preprocessing import detect_types, normalize_and_remap, denormalize_and_remap


def missforest(X: pd.DataFrame, cat_cols, max_iter=10, n_estimators=100, seed=0):
    """MissForest (Stekhoven & Buehlmann 2012): mean/mode initialisation, then
    iteratively re-impute each column with a random forest (classifier for
    categorical, regressor for numerical) in order of increasing missingness,
    stopping when the change in the imputed matrix increases."""
    X = X.copy()
    miss = X.isna()
    cols = [c for c in miss.sum().sort_values(kind="stable").index
            if miss[c].any() and not miss[c].all()]
    for c in X.columns:
        if X[c].isna().all():
            X[c] = 0.0  # column entirely missing: nothing to learn from
        elif c in cat_cols:
            X[c] = X[c].fillna(X[c].mode().iloc[0])
        else:
            X[c] = X[c].fillna(X[c].mean())
    num_cols = [c for c in X.columns if c not in cat_cols]
    prev_diff_n, prev_diff_c, prev = np.inf, np.inf, X.copy()
    for _ in range(max_iter):
        for c in cols:
            obs, mis = ~miss[c], miss[c]
            others = X.drop(columns=c)
            Est = RandomForestClassifier if c in cat_cols else RandomForestRegressor
            est = Est(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
            y = X.loc[obs, c].astype(int) if c in cat_cols else X.loc[obs, c]
            est.fit(others[obs], y)
            X.loc[mis, c] = est.predict(others[mis])
        dn = (((X[num_cols] - prev[num_cols]) ** 2).to_numpy().sum()
              / max((X[num_cols] ** 2).to_numpy().sum(), 1e-12)) if num_cols else 0
        dc = ((X[cat_cols] != prev[cat_cols]).to_numpy().sum()
              / max(miss[cat_cols].to_numpy().sum(), 1)) if cat_cols else 0
        if dn >= prev_diff_n and dc >= prev_diff_c:
            X = prev  # stopping criterion: return previous iteration
            break
        prev_diff_n, prev_diff_c, prev = dn, dc, X.copy()
    return X


def _round_cats(imp: pd.DataFrame, ref: pd.DataFrame, cat_cols):
    for c in cat_cols:
        lo, hi = ref[c].min(), ref[c].max()
        imp[c] = imp[c].round().clip(lo, hi)
    return imp


def impute_baseline(name: str, df_with_na: pd.DataFrame, distinct_ratio_threshold=0.01,
                    seed=0, fit_rows=None) -> pd.DataFrame:
    """Return the completed frame in the original format and row order.

    `fit_rows` (optional boolean mask) restricts the rows the imputer sees, e.g.
    only the incomplete test rows (ablation); other rows are returned unchanged.
    """
    types = detect_types(df_with_na, distinct_ratio_threshold)
    cat, num, dt, cat_u = (types["categorical"], types["numerical"],
                           types["datetime"], types["categorical_unique"])
    norm = normalize_and_remap(df_with_na, cat, num, dt, cat_u)
    cat_all = cat + cat_u
    full = norm
    if fit_rows is not None:
        norm = norm[np.asarray(fit_rows, bool)]

    if name == "MeanMode":
        imp = norm.copy()
        for c in imp.columns:
            if imp[c].isna().all():
                imp[c] = 0.0
                continue
            fill = imp[c].mode().iloc[0] if c in cat_all else imp[c].mean()
            imp[c] = imp[c].fillna(fill)
    elif name == "MissForest":
        imp = missforest(norm, cat_all, seed=seed)
    else:
        if name == "KNNImputer":
            est = KNNImputer(n_neighbors=5, keep_empty_features=True)
        elif name == "IterativeImputer":
            est = IterativeImputer(max_iter=10, random_state=seed, keep_empty_features=True)
        elif name == "MICE":
            # chained equations with posterior sampling (one stochastic draw)
            est = IterativeImputer(max_iter=10, sample_posterior=True, random_state=seed,
                                   keep_empty_features=True)
        else:
            raise ValueError(name)
        imp = pd.DataFrame(est.fit_transform(norm), columns=norm.columns, index=norm.index)
        imp = _round_cats(imp, norm, cat_all)
        for c in num:  # keep numericals in the observed [0,1] range
            imp[c] = imp[c].clip(0, 1)
    if fit_rows is not None:
        full = full.copy()
        full.loc[imp.index] = imp
        imp = full
    imp = imp.reset_index(drop=True)
    return denormalize_and_remap(imp, df_with_na, cat, num, dt, cat_u)
