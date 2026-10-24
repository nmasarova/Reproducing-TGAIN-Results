"""Data-type detection and (de)normalisation (thesis Section 4.1).

Port of `type_determinator`, `normalize_and_remap` and `denormalize_and_remap`
from the original `UtilFunctions.py`, updated for pandas >= 2 / numpy >= 2.
Behaviour is kept identical to the original unless stated in a comment marked
`# CHANGE:`.
"""
from __future__ import annotations

import datetime
from collections import Counter

import numpy as np
import pandas as pd

_DT_FORMATS = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f")


def _is_datetime(value) -> bool:
    if not isinstance(value, str):
        return False
    for fmt in _DT_FORMATS:
        try:
            datetime.datetime.strptime(value, fmt)
            return True
        except ValueError:
            pass
    return False


def type_determinator(column_values: pd.Series, distinct_ratio_threshold: float) -> str:
    """Return 'datetime', 'categorical', 'categorical_unique' or 'numerical'.

    Rules (identical to the original):
    * every value parses as a date            -> 'datetime'
    * contains strings and #unique <= thr*#rows -> 'categorical'
    * contains strings and #unique >  thr*#rows -> 'categorical_unique'
    * numeric: distinct ratio (with or without the most common value) < thr
                                                -> 'categorical', else 'numerical'
    """
    # CHANGE: the original returned 'datetime' as soon as the *first* value parsed
    # as a date (the return statement sat inside the loop). We require all
    # non-missing values to parse. Irrelevant for the German data (no dates).
    non_null = column_values.dropna()
    if len(non_null) > 0 and all(_is_datetime(v) for v in non_null):
        return "datetime"

    has_str = any(isinstance(v, str) for v in column_values)
    n_unique = len(column_values.unique())  # NaN counts as a value, as in the original
    if has_str:
        if n_unique <= distinct_ratio_threshold * len(column_values):
            return "categorical"
        return "categorical_unique"

    values = np.asarray(non_null)
    most_common = Counter(values).most_common(1)[0][0]
    without_mc = values[values != most_common]
    if len(without_mc) != 0:
        if len(np.unique(without_mc)) / len(without_mc) < distinct_ratio_threshold:
            return "categorical"
    if len(np.unique(values)) / len(values) < distinct_ratio_threshold:
        return "categorical"
    return "numerical"


def detect_types(df: pd.DataFrame, distinct_ratio_threshold: float) -> dict[str, list[str]]:
    groups = {"categorical": [], "numerical": [], "datetime": [], "categorical_unique": []}
    for col in df.columns:
        groups[type_determinator(df[col], distinct_ratio_threshold)].append(col)
    return groups


def normalize_and_remap(df, categorical, numerical, datetime_cols, categorical_unique):
    """Categoricals -> integer codes (sorted category order), numericals -> [0,1]
    min-max, datetimes -> int64 nanoseconds. Missing values stay NaN."""
    out = df.copy()
    for col in list(categorical) + list(categorical_unique):
        codes = out[col].astype("category").cat.codes.astype(float)
        out[col] = codes.replace({-1.0: np.nan})
    for col in numerical:
        mn, mx = out[col].min(), out[col].max()
        out[col] = (out[col].astype(float) - mn) / (mx - mn)
    for col in datetime_cols:
        conv = pd.to_datetime(out[col]).values.astype(np.int64).astype(float)
        out[col] = np.where(conv == np.iinfo(np.int64).min, np.nan, conv)
    return out


def denormalize_and_remap(norm_df, original_df, categorical, numerical, datetime_cols,
                          categorical_unique):
    """Inverse of `normalize_and_remap`, using the original (incomplete) frame to
    recover category labels and min/max."""
    out = norm_df.copy()
    for col in list(categorical) + list(categorical_unique):
        cats = original_df[col].astype("category").cat.categories
        out[col] = [np.nan if pd.isna(v) else cats[int(v)] for v in out[col]]
    for col in numerical:
        orig = original_df[col]
        mn, mx = orig.min(), orig.max()
        vals = out[col] * (mx - mn) + mn
        # Original: cast to int (truncation, not rounding!) unless the original
        # column contained floats. A column with NaN is float in pandas, so in
        # practice columns with missing values stay float.
        is_float = any(isinstance(v, float) for v in orig)
        out[col] = vals.astype(float) if is_float else vals.astype(int)
    for col in datetime_cols:
        conv = pd.to_datetime(out[col])
        out[col] = [np.nan if pd.isna(t) else str(t)[:10] for t in conv]
    return out
