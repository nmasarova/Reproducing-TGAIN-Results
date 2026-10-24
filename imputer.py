"""TGAIN imputation (thesis Section 4.3, Algorithm 1) and the full framework
(`data_imputation_completer`, Fig. 4.1). Port of the original `imputer()` in
`UtilFunctions.py` and `data_imputation_completer()` in `tabularGAIN.py`.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .model import TabularCTGAN
from .preprocessing import detect_types, normalize_and_remap, denormalize_and_remap


def impute_rows(model: TabularCTGAN, rows_to_impute: pd.DataFrame, categorical_columns,
                n_samples=50, max_tries=50, rng: random.Random | None = None) -> pd.DataFrame:
    """Algorithm 1: for each incomplete row
    1. condition on all observed *categorical* columns and draw up to 50 rows;
    2. if no row can be drawn, drop a random condition and retry (until none left);
    3. rank synthetic rows by Euclidean distance on the observed non-categorical
       columns; take the 3 closest (alpha, beta, gamma);
    4. for every missing cell: beta if alpha != beta and beta == gamma, else alpha.
       (fewer than 3 rows -> use the closest one).
    """
    rng = rng or random.Random()
    out = rows_to_impute.copy()
    columns = list(out.columns)
    for idx, full_row in rows_to_impute.iterrows():
        observed = full_row.dropna().to_dict()
        conditions = {k: v for k, v in observed.items() if k in categorical_columns}
        compare = {k: v for k, v in observed.items() if k not in categorical_columns}
        missing_cols = [c for c in columns if pd.isna(full_row[c])]
        if not missing_cols:
            continue
        while True:
            try:
                if len(conditions) == 0:
                    new_data = model.sample(n_samples).reset_index(drop=True)
                else:
                    new_data = model.sample_conditions(conditions, num_rows=n_samples,
                                                       max_tries=max_tries)
            except ValueError:
                # drop one random condition and retry
                conditions.pop(rng.choice(list(conditions.keys())))
                continue
            break

        if compare:
            target = np.array(list(compare.values()), dtype=float)
            synth = new_data[list(compare.keys())].to_numpy(dtype=float)
            dist = np.sqrt(((synth - target) ** 2).sum(axis=1))
        else:
            dist = np.zeros(len(new_data))
        order = np.argsort(dist, kind="stable")
        if len(order) < 3:
            best = new_data.iloc[order[0]]
            for c in missing_cols:
                out.at[idx, c] = best[c]
        else:
            a, b, g = (new_data.iloc[order[i]] for i in range(3))
            for c in missing_cols:
                out.at[idx, c] = b[c] if (a[c] != b[c] and b[c] == g[c]) else a[c]
    return out


def majority_vote(rows_with_missing, candidates):
    """Combine the imputations of 3 models (Fig. 4.5): all equal or any pair equal ->
    majority value; all different -> model 1."""
    m1, m2, m3 = candidates
    out = rows_with_missing.copy()
    for idx, row in rows_with_missing.iterrows():
        for c in rows_with_missing.columns:
            if pd.isna(row[c]):
                v1, v2, v3 = m1.at[idx, c], m2.at[idx, c], m3.at[idx, c]
                out.at[idx, c] = v2 if (v2 == v3 and v1 != v2) else v1
    return out


def data_imputation_completer(df_with_na: pd.DataFrame, nr_epochs=300,
                              distinct_ratio_threshold=0.1, generator_lr=2e-4,
                              discriminator_lr=2e-4, verbose=False, n_models=3,
                              n_samples=50, max_tries=50, discrete_mode="original",
                              seed=None, pool_size=50_000, return_details=False,
                              preserve_order=False):
    """Full TGAIN pipeline. Returns the completed frame in the original format
    (complete rows first, then the imputed rows - as in the original code).
    With `preserve_order=True` the rows keep the input order instead (needed to
    evaluate correctly when some test rows happen to have no missing value).

    With `return_details=True` also returns a dict with the completed frames of
    each individual model (so the 1-model variant comes for free) and metadata.
    """
    if not isinstance(df_with_na, pd.DataFrame) or df_with_na.empty:
        raise ValueError("Input needs to be a non-empty pandas DataFrame")

    types = detect_types(df_with_na, distinct_ratio_threshold)
    cat, num, dt, cat_u = (types["categorical"], types["numerical"],
                           types["datetime"], types["categorical_unique"])
    norm = normalize_and_remap(df_with_na, cat, num, dt, cat_u)

    rows_missing = norm[norm.isnull().any(axis=1)]
    complete = norm.dropna()
    if len(rows_missing) == 0 or len(complete) == 0:
        raise ValueError("Need both complete rows (training) and incomplete rows")

    rng = random.Random(seed)
    per_model, models = [], []
    for m in range(n_models):
        mseed = None if seed is None else seed * 1000 + m
        model = TabularCTGAN(epochs=nr_epochs, generator_lr=generator_lr,
                             discriminator_lr=discriminator_lr, verbose=verbose,
                             discrete_columns=cat, discrete_mode=discrete_mode,
                             seed=mseed, pool_size=pool_size)
        model.fit(complete)
        models.append(model)
        per_model.append(impute_rows(model, rows_missing, cat, n_samples, max_tries, rng))

    imputed = majority_vote(rows_missing, per_model) if n_models == 3 else per_model[0]

    def finish(imp):
        total = pd.concat([complete, imp], sort=False)
        total = total.sort_index() if preserve_order else total
        total = total.reset_index(drop=True)
        return denormalize_and_remap(total, df_with_na, cat, num, dt, cat_u)

    result = finish(imputed)
    if not return_details:
        return result
    details = {"types": types, "per_model": [finish(p) for p in per_model],
               "losses": [m.loss_values() for m in models]}
    return result, details
