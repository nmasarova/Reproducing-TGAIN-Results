"""Custom prediction-error metric (thesis Section 5.2.1).

Two versions are provided:

`evaluate_imputations_original`  - faithful port of the original
    `evaluate_imputations()` in `UtilFunctions.py`; this is the code that
    produced the numbers in the thesis/notebook. It has some quirks:
      (q1) the true and the imputed test set are normalised/label-encoded
           *independently* (each with its own min/max and its own category list);
      (q2) the final score is (num_err + alpha*cat_err)/2, whereas the formula
           printed in the thesis weights by #numeric/#total and #categoric/#total;
      (q3) `calc_avg_dissimilarity` skips the last element (off-by-one).

`evaluate_imputations_corrected` - same idea, but both frames are encoded with
    a shared encoding (categories and min/max from the full data set), the
    pairwise dissimilarity is computed over all pairs, and the thesis formula
    weighting (#numeric/#total, #categoric/#total) is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from .preprocessing import detect_types, normalize_and_remap


def _calc_avg_dissimilarity_original(values):
    values = list(values)
    dis = sim = 0
    for idx, v in enumerate(values):
        for i in range(idx + 1, len(values) - 1):  # (q3) off-by-one kept
            if v == values[i]:
                sim += 1
            else:
                dis += 1
    return dis / (dis + sim)


def _calc_avg_dissimilarity(values):
    v = np.asarray(values)
    eq = (v[:, None] == v[None, :])
    n = len(v)
    iu = np.triu_indices(n, 1)
    return 1.0 - eq[iu].mean()


def _alpha(true_norm, num_vars, cat_vars, dissim):
    num_d = [np.average(pairwise_distances(true_norm[v].to_numpy().reshape(-1, 1)))
             for v in num_vars]
    cat_d = [dissim(true_norm[v]) for v in cat_vars]
    return np.mean(num_d) / np.mean(cat_d)


def _errors(true_norm, imp_norm, mask, num_vars, cat_vars):
    cols = list(true_norm.columns)
    num_idx = [cols.index(c) for c in num_vars]
    cat_idx = [cols.index(c) for c in cat_vars]
    t, p = true_norm.to_numpy(dtype=float), imp_norm.to_numpy(dtype=float)
    mn = np.zeros_like(mask); mn[:, num_idx] = mask[:, num_idx]
    mc = np.zeros_like(mask); mc[:, cat_idx] = mask[:, cat_idx]
    num_err = np.abs(t[mn] - p[mn]).mean() if mn.any() else np.nan
    cat_err = (t[mc] != p[mc]).mean() if mc.any() else np.nan
    return num_err, cat_err, int(mn.sum()), int(mc.sum())


def evaluate_imputations_original(train, test_complete, mask, test_imputed, threshold,
                                  return_parts=False):
    types = detect_types(train, threshold)
    cat, num, dt, cat_u = (types["categorical"], types["numerical"],
                           types["datetime"], types["categorical_unique"])
    t = normalize_and_remap(test_complete.copy(), cat, num, dt, cat_u)   # (q1)
    p = normalize_and_remap(test_imputed.copy(), cat, num, dt, cat_u)    # (q1)
    for col in dt:
        mn, mx = t[col].min(), t[col].max()
        t[col] = (t[col] - mn) / (mx - mn)
        p[col] = (p[col] - mn) / (mx - mn)
    num_vars, cat_vars = num + dt, cat + cat_u
    alpha = _alpha(t, num_vars, cat_vars, _calc_avg_dissimilarity_original)
    num_err, cat_err, n_num, n_cat = _errors(t, p, np.asarray(mask, bool), num_vars, cat_vars)
    score = (num_err + alpha * cat_err) / 2                              # (q2)
    if return_parts:
        return score, dict(num_err=num_err, cat_err=cat_err, alpha=alpha,
                           n_num=n_num, n_cat=n_cat)
    return score


def evaluate_imputations_corrected(full_df, test_complete, mask, test_imputed, threshold,
                                   return_parts=False):
    """`full_df` (e.g. the whole data set) defines the shared encoding."""
    types = detect_types(full_df, threshold)
    cat, num, dt, cat_u = (types["categorical"], types["numerical"],
                           types["datetime"], types["categorical_unique"])
    n_t = len(test_complete)
    stacked = pd.concat([full_df, test_complete, test_imputed], ignore_index=True)
    enc = normalize_and_remap(stacked, cat, num, dt, cat_u)
    for col in dt:
        mn, mx = enc[col].min(), enc[col].max()
        enc[col] = (enc[col] - mn) / (mx - mn)
    t = enc.iloc[len(full_df):len(full_df) + n_t].reset_index(drop=True)
    p = enc.iloc[len(full_df) + n_t:].reset_index(drop=True)
    num_vars, cat_vars = num + dt, cat + cat_u
    alpha = _alpha(t, num_vars, cat_vars, _calc_avg_dissimilarity)
    num_err, cat_err, n_num, n_cat = _errors(t, p, np.asarray(mask, bool), num_vars, cat_vars)
    n_tot = len(num_vars) + len(cat_vars)
    score = len(num_vars) / n_tot * num_err + len(cat_vars) / n_tot * alpha * cat_err
    if return_parts:
        return score, dict(num_err=num_err, cat_err=cat_err, alpha=alpha,
                           n_num=n_num, n_cat=n_cat)
    return score
