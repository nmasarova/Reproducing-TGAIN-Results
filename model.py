"""CTGAN wrapper that reproduces the behaviour of `sdv.tabular.CTGAN` (sdv 0.15)
as it was used by the original TGAIN code, on top of the maintained `ctgan`
package (no SDV dependency).

What sdv 0.15 did with the TGAIN input
--------------------------------------
TGAIN passes a fully numeric (float) frame to `sdv.tabular.CTGAN` without
metadata. sdv 0.15 therefore:
  1. inferred every float column as *numerical* -> the integer category codes
     were modelled as continuous columns by CTGAN (mode-specific normalisation);
  2. on sampling, rounded each column to the number of decimals seen in the
     training data (`rounding='auto'`) and clipped to the training min/max
     (`min_value='auto'`, `max_value='auto'`);
  3. implemented `sample_conditions` by *rejection sampling*: CTGAN in sdv 0.15
     raised NotImplementedError for conditional sampling, so sdv drew batches of
     `10 * num_rows` unconditional rows, kept rows matching all conditions, and
     stopped after `num_rows` valid rows or `max_tries` batches. If no row
     matched, it raised `ValueError`.

`discrete_mode='original'` (default) emulates 1-2; `discrete_mode='discrete'`
declares the categorical codes as discrete columns to CTGAN instead (closer to
how the thesis text describes CTGAN, offered as an ablation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN


def _decimals(series: pd.Series, max_decimals: int = 15) -> int | None:
    """Number of decimals needed to represent all values (None = no rounding)."""
    vals = series.dropna().to_numpy(dtype=float)
    for d in range(max_decimals + 1):
        if np.allclose(vals, np.round(vals, d), rtol=0, atol=0):
            return d
    return None


class TabularCTGAN:
    def __init__(self, epochs=300, generator_lr=2e-4, discriminator_lr=2e-4,
                 batch_size=500, discriminator_steps=1, pac=10,
                 embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_decay=1e-6, discriminator_decay=0.0, log_frequency=True,
                 discrete_columns=(), discrete_mode="original", verbose=False,
                 seed=None, pool_size=50_000, enable_gpu=False):
        self.kw = dict(epochs=epochs, generator_lr=generator_lr,
                       discriminator_lr=discriminator_lr, batch_size=batch_size,
                       discriminator_steps=discriminator_steps, pac=pac,
                       embedding_dim=embedding_dim, generator_dim=generator_dim,
                       discriminator_dim=discriminator_dim, generator_decay=generator_decay,
                       discriminator_decay=discriminator_decay,
                       log_frequency=log_frequency, verbose=verbose,
                       enable_gpu=enable_gpu)
        self.discrete_columns = list(discrete_columns)
        self.discrete_mode = discrete_mode
        self.seed = seed
        self.pool_size = pool_size
        self._pool = None
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ fit
    def fit(self, data: pd.DataFrame):
        self.columns = list(data.columns)
        data = data.astype(float)
        self.min_ = data.min()
        self.max_ = data.max()
        self.decimals_ = {c: _decimals(data[c]) for c in self.columns}
        if self.discrete_mode == "discrete":
            disc = [c for c in self.discrete_columns if c in self.columns]
            data = data.copy()
            for c in disc:
                data[c] = data[c].astype(int).astype(str)
        else:
            disc = []
        self._disc = disc
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        self.model = CTGAN(**self.kw)
        self.model.fit(data, discrete_columns=disc)
        self._pool = None
        return self

    def loss_values(self):
        return getattr(self.model, "loss_values", None)

    # --------------------------------------------------------------- sample
    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self._disc:
            df[c] = df[c].astype(float)
        for c in self.columns:
            col = df[c].astype(float).clip(self.min_[c], self.max_[c])
            d = self.decimals_[c]
            df[c] = col.round(d) if d is not None else col
        return df[self.columns]

    def sample(self, num_rows: int) -> pd.DataFrame:
        if self.seed is not None:
            torch.manual_seed(int(self.rng.integers(2**31)))
        return self._postprocess(self.model.sample(num_rows))

    def _get_pool(self) -> pd.DataFrame:
        """Generated rows are i.i.d., so instead of calling the generator for every
        batch we draw once a large pool and sub-sample it (pure speed-up)."""
        if self._pool is None:
            self._pool = self.sample(self.pool_size).reset_index(drop=True)
            self._pool_np = self._pool.to_numpy()
        return self._pool

    def sample_conditions(self, conditions: dict, num_rows=50, max_tries=50,
                          batch_size_per_try=None) -> pd.DataFrame:
        """Rejection sampling as in sdv 0.15 `sample_conditions`.

        Draw up to `max_tries` batches of `batch_size_per_try` (default
        10*num_rows) rows; keep rows whose values equal all conditions; stop at
        `num_rows` valid rows. Raises ValueError if none is valid.
        """
        batch = batch_size_per_try or 10 * num_rows
        n_cand = batch * max_tries
        if self.pool_size is None:
            cand = self.sample(n_cand).reset_index(drop=True)
            cand_np = cand.to_numpy()
        else:
            pool = self._get_pool()
            idx = self.rng.choice(len(pool), size=min(n_cand, len(pool)), replace=False)
            cand = pool.iloc[idx].reset_index(drop=True)
            cand_np = self._pool_np[idx]
        mask = np.ones(len(cand), dtype=bool)
        for col, val in conditions.items():
            mask &= cand_np[:, self.columns.index(col)] == val
        valid = cand[mask]
        if len(valid) == 0:
            raise ValueError("Unable to sample any rows for the given conditions.")
        return valid.iloc[:num_rows].reset_index(drop=True)
