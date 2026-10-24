# Reproducing TGAIN: GAN-based imputation for mixed-type tabular data

This repo reproduces the main experiment of

> O. Bannany (2022). *Imputing missing values for mixed-type tabular data sets using generative adversarial networks.* MSc thesis, Utrecht University. Supervisors: H. Qahtan and M. W. Chekol.
> Original code and thesis PDF: <https://github.com/hqahtan/GANS_For_Data_Imputation> (`TGAIN_illustration.ipynb`, `tabularGAIN.py`, `UtilFunctions.py`, `Thesis_Ouassim.pdf`).

It contains:

1. a **re-implementation of TGAIN for current Python (3.10–3.13)**,
2. the **main experiment from the thesis (Table 5.2, German Credit)**, repeated over several random seeds, with all four baselines,
3. a comparison with the published numbers and an explanation of the differences,
4. a check against the **unmodified original code**, which was run in a rebuilt legacy environment (Python 3.8, sdv 0.15).

---

## 1. Why a re-implementation was needed

The original code doesn't run on a current Python stack:

| Problem | Details |
|---|---|
| Retired Python / libraries | It was tested on Python 3.7 (Colab, 2022) with `sdv==0.15.0`, `rdt==0.6.4`, `pandas==1.3.5`, `numpy==1.21.6`. Python 3.7 is end-of-life. The `sdv.tabular` module was removed in SDV 1.0, so `from sdv.tabular import CTGAN` and `model.sample_conditions(...)` no longer exist. |
| Missing data file | The notebook loads `rawData/german/german_data_preprocessed.csv`. This file isn't in the public repo, and the repo the README points to is no longer public. |
| Pandas 2 behaviour change | In pandas ≥ 2, the category label `"None"` (used for one *Credit_history* value) is read as missing. |

Fixes in this repo:

* **`tgain/`** is a line-by-line port of the original algorithm. It uses the maintained MIT-licensed [`ctgan`](https://github.com/sdv-dev/CTGAN) package instead of SDV.
* **`tgain/model.py` reproduces how `sdv.tabular.CTGAN` 0.15 actually behaved on TGAIN's input.** I confirmed this by running sdv 0.15 itself (see §6):
  * every column, including the integer category codes, was treated as *numerical* by CTGAN;
  * samples were rounded to the training data's number of decimals and clipped to the training min/max;
  * `sample_conditions` was pure rejection sampling: batches of `10 × num_rows` rows, at most `max_tries` batches, `ValueError` if no row matched.
* **`data/prepare_german.py`** rebuilds the missing CSV from the UCI data. It matches every property reported in the thesis (see §3).

## 2. Repository layout

```
tgain/
  preprocessing.py   type detection (distinct-ratio rule), min-max normalisation, label encoding
  model.py           CTGAN wrapper emulating sdv 0.15 (rounding, clipping, rejection sampling)
  imputer.py         Algorithm 1 (per-row imputation) + 3-model majority vote + full pipeline
  evaluation.py      thesis prediction-error metric: faithful port + corrected version
  baselines.py       MissForest, MICE, KNNImputer, IterativeImputer (+ Mean/Mode reference)
data/prepare_german.py         rebuilds german_data_preprocessed.csv from UCI
experiments/run_german.py      main experiment (Table 5.2), resumable, writes results/*.csv
experiments/make_report.py     tables (results/summary.md) + figures
notebooks/TGAIN_reproduction.ipynb   notebook version (local or Colab)
legacy/run_original_code.py    runs the UNMODIFIED original code (Python 3.8 env)
results/                       all raw results, summary.md, figures
```

## 3. Quick start

```bash
python -m venv .venv && source .venv/bin/activate          # Python 3.10–3.13
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python data/prepare_german.py                    # -> data/german_data_preprocessed.csv
python experiments/run_german.py --reps 6        # ~1 h on a 2-core CPU
python experiments/run_german.py --reps 3 --no-baselines --discrete-mode discrete \
       --out results/german_results_discrete.csv  # optional ablation
python experiments/make_report.py                # -> results/summary.md + PNGs
```

Using the imputer on your own data:

```python
from tgain import data_imputation_completer
completed = data_imputation_completer(df_with_nans, nr_epochs=300, distinct_ratio_threshold=0.01,
                                      generator_lr=2e-4, discriminator_lr=2e-4, seed=0)
```

**Rebuilding the dataset.** `prepare_german.py` reproduces every reported property of the thesis data:

* 999 rows: the first UCI record was lost because the file was read without `header=None`;
* 21 columns: 3 continuous and 18 categorical;
* 432 complete and 567 incomplete rows;
* the 8 missingness-pattern proportions of thesis Fig. 5.1 match exactly (0.432 / 0.263 / 0.090 / 0.083 / 0.067 / 0.032 / 0.016 / 0.016).

The three columns with missing values are the UCI "unknown/none" codes:

* `Status_existing_account = A14`
* `Savings = A65`
* `Property = A124`

## 4. The algorithm

1. **Detect column types.** Columns are classified as datetime, categorical, categorical_unique or numerical using the *distinct-ratio threshold* (Eq. 4.1).
2. **Normalise.** Numerical columns are min-max scaled to [0, 1]. Categorical columns are mapped to integer codes.
3. **Split the data.** Complete rows become the training set. Rows with at least one missing value are the rows to impute.
4. **Train three CTGAN models** independently on the complete rows.
5. **Impute each incomplete row with each model (Algorithm 1).**
   * Draw up to 50 synthetic rows conditioned on the row's observed *categorical* values, using rejection sampling.
   * If no row can be drawn, remove a random condition and try again.
   * Rank the synthetic rows by Euclidean distance on the observed non-categorical columns.
   * Fill each missing cell with a vote among the 3 closest rows: β if α ≠ β = γ, otherwise α.
6. **Combine the models.** Take the majority vote across the 3 models. If all three disagree, use model 1. The "TGAIN (1 model)" variant uses model 1 alone.
7. **Denormalise** and map the codes back to category labels.

## 5. Parameters used

### TGAIN

| Parameter | Value | Source |
|---|---|---|
| `nr_epochs` | 300 | notebook / thesis §4.2 |
| `distinct_ratio_threshold` | 0.01 | notebook (README default is 0.1) |
| `generator_lr`, `discriminator_lr` | 2e-4, 2e-4 | notebook / thesis |
| number of GAN models | 3 (and 1 for the variant) | thesis |
| synthetic rows per condition (`num_rows`) | 50 | original `imputer()` |
| `max_tries` | 50 | original `imputer()` |
| `batch_size_per_try` | 500 (= 10 × num_rows, sdv 0.15 default) | sdv 0.15 |
| candidates in the per-model vote | top-3 by Euclidean distance | thesis Alg. 1 |
| CTGAN: `embedding_dim`, `generator_dim`, `discriminator_dim` | 128, (256, 256), (256, 256) | sdv 0.15 / ctgan defaults |
| CTGAN: `batch_size`, `pac`, `discriminator_steps` | 500, 10, 1 | defaults |
| CTGAN: `generator_decay`, `discriminator_decay`, `log_frequency` | 1e-6, 0, True | defaults |
| column treatment in CTGAN (`discrete_mode`) | `original`: all columns continuous, then rounded and clipped | emulates sdv 0.15 |
| `pool_size` (implementation detail) | 50 000 pre-generated rows per model. Rejection sampling draws its candidates from this pool. This only speeds things up, because generated rows are i.i.d. | this repo |

### Experiment protocol (same as `TGAIN_illustration.ipynb`)

| Parameter | Value |
|---|---|
| Data | German Credit, preprocessed: 999 rows, 21 columns |
| Train / test | 432 complete rows, `shuffle(random_state=0)`; first 50 = test, remaining 382 = train |
| Missingness | MCAR on the 50 test rows: each cell (all 21 columns, target included) removed with probability *p* |
| Missing rates *p* | 0.1, 0.3, 0.5, 0.7, 0.9 |
| Repetitions | **6** per rate. The thesis reports one unseeded run. |
| Seeds | mask: `RandomState(100·p·1000 + rep)`; TGAIN: `rep·100 + 100·p` (model *m* gets `seed·1000+m`) |
| Evaluation | thesis prediction error (§5.2.1), threshold 0.01; computed on the 50 test rows |

### Baselines

The thesis doesn't publish the baseline code or settings, so these are my choices.

| Baseline | Implementation |
|---|---|
| MissForest | own implementation of Stekhoven & Bühlmann: RF classifier for categoricals, RF regressor for numericals, 100 trees, ≤ 10 iterations, original stopping rule |
| MICE | `IterativeImputer(sample_posterior=True, max_iter=10)` (BayesianRidge) |
| IterativeImputer | `IterativeImputer(max_iter=10)` (BayesianRidge, posterior mean) |
| KNNImputer | `KNNImputer(n_neighbors=5)` |
| Mean/Mode (extra, not in the thesis) | column mean for numericals, mode for categoricals |

All baselines run on the same normalised, label-encoded matrix as TGAIN (382 training rows + 50 test rows). Imputed category codes are rounded and clipped.

### Software

* **Modern environment:** Python 3.13, torch 2.14 (CPU), ctgan 0.12.1, rdt 2.0, pandas 2.3, numpy 2.5, scikit-learn 1.9.
* **Legacy check:** Python 3.8.20, sdv 0.15.0, rdt 0.6.4, torch 1.13.1, pandas 1.3.5, numpy 1.21.6 (`legacy/requirements-legacy.txt`).

## 6. Is the port correct?

| Check | Result |
|---|---|
| Evaluation metric: original `evaluate_imputations` (py3.8) vs `evaluate_imputations_original` (py3.13), same inputs, 4 cases | identical to 10 decimals |
| Imputation logic: original `imputer()` vs `impute_rows()`, both driven by the same deterministic fake generator | identical except 13 of 1050 cells. In those rows several synthetic rows tie on distance, and the two versions sort ties differently. |
| sdv 0.15 behaviour assumed in `model.py` | confirmed by running sdv 0.15: all 21 columns inferred as `numerical/float`, samples rounded to integers, rejection sampling raises `ValueError` when nothing matches |
| **Unmodified original code, end-to-end, same split and mask** | 30 %: **0.2987** (original) vs **0.2995** (port, rep 0). 50 %: 0.2871 vs 0.2730. Both within the seed-to-seed spread (sd ≈ 0.03). |

The original pipeline needs ~1 h per run because SDV's sampler is slow. The port needs ~1 min.

## 7. Results

### 7.1 Thesis (Table 5.2, single run)

| Method | 10% | 30% | 50% | 70% | 90% |
|---|---|---|---|---|---|
| TGAIN (1 model) | 0.2487 | 0.2543 | 0.2557 | 0.3176 | 0.3197 |
| TGAIN (3 models) | 0.2416 | 0.2605 | 0.2685 | 0.2941 | 0.2948 |
| MissForest | 0.3250 | 0.3187 | 0.3140 | 0.3527 | 0.3980 |
| MICE | 0.2840 | 0.3140 | 0.3054 | 0.3084 | 0.3002 |
| KNNImputer | 0.2525 | 0.2941 | 0.2941 | 0.3290 | 0.3155 |
| IterativeImputer | 0.3335 | 0.3368 | 0.3581 | 0.3572 | 0.3412 |

### 7.2 Reproduction: same metric as the thesis (mean ± sd over 6 repetitions)

| Method | 10% | 30% | 50% | 70% | 90% |
|---|---|---|---|---|---|
| TGAIN (1 model) | 0.2935 ± 0.0208 | 0.3071 ± 0.0302 | 0.3005 ± 0.0251 | 0.3096 ± 0.0237 | 0.3346 ± 0.0181 |
| TGAIN (3 models) | 0.2908 ± 0.0262 | 0.3079 ± 0.0309 | 0.2937 ± 0.0232 | 0.3111 ± 0.0264 | 0.3401 ± 0.0234 |
| MissForest | **0.1758** ± 0.0256 | **0.2159** ± 0.0223 | **0.2438** ± 0.0133 | **0.2831** ± 0.0231 | 0.3538 ± 0.0232 |
| MICE | 0.2689 ± 0.0228 | 0.3068 ± 0.0128 | 0.3058 ± 0.0127 | 0.3379 ± 0.0144 | 0.3460 ± 0.0105 |
| KNNImputer | 0.2393 ± 0.0194 | 0.2522 ± 0.0215 | 0.2633 ± 0.0118 | 0.3070 ± 0.0199 | 0.3439 ± 0.0073 |
| IterativeImputer | 0.2350 ± 0.0183 | 0.2517 ± 0.0097 | 0.2587 ± 0.0107 | 0.3094 ± 0.0145 | 0.3891 ± 0.0115 |
| Mean/Mode (reference) | 0.2460 ± 0.0294 | 0.2577 ± 0.0161 | 0.2767 ± 0.0142 | 0.3239 ± 0.0210 | 0.4296 ± 0.0193 |

![Thesis vs reproduction](results/german_error_original.png)

Mean rank of each method across repetitions (1 = best), with the thesis rank in brackets:

| Method | 10% | 30% | 50% | 70% | 90% |
|---|---|---|---|---|---|
| TGAIN (1 model) | 5.5 (2) | 5.2 (1) | 5.2 (1) | 3.3 (3) | 2.0 (4) |
| TGAIN (3 models) | 5.3 (1) | 4.8 (2) | 4.3 (2) | 4.2 (1) | 3.0 (1) |
| MissForest | 1.0 (5) | 1.0 (5) | 1.2 (5) | 1.0 (5) | 3.8 (6) |
| MICE | 3.8 (4) | 5.0 (4) | 5.2 (4) | 5.3 (2) | 3.3 (2) |
| KNNImputer | 2.8 (3) | 2.7 (3) | 3.2 (3) | 3.5 (4) | 3.0 (3) |
| IterativeImputer | 2.5 (6) | 2.3 (6) | 2.0 (6) | 3.7 (6) | 5.8 (5) |

### 7.3 Reproduction: corrected metric

This version uses a shared encoding, all pairs in the dissimilarity, and the thesis formula weights (see §8.3).

| Method | 10% | 30% | 50% | 70% | 90% |
|---|---|---|---|---|---|
| TGAIN (1 model) | 0.1455 ± 0.0129 | 0.1450 ± 0.0104 | 0.1425 ± 0.0076 | 0.1533 ± 0.0067 | 0.1564 ± 0.0052 |
| TGAIN (3 models) | 0.1397 ± 0.0173 | 0.1457 ± 0.0103 | 0.1362 ± 0.0086 | 0.1447 ± 0.0055 | 0.1472 ± 0.0038 |
| MissForest | **0.0877** ± 0.0093 | **0.0975** ± 0.0073 | **0.1069** ± 0.0097 | **0.1259** ± 0.0027 | 0.1377 ± 0.0040 |
| MICE | 0.1452 ± 0.0082 | 0.1568 ± 0.0073 | 0.1559 ± 0.0076 | 0.1694 ± 0.0049 | 0.1750 ± 0.0043 |
| KNNImputer | 0.1273 ± 0.0075 | 0.1302 ± 0.0102 | 0.1341 ± 0.0074 | 0.1498 ± 0.0033 | 0.1502 ± 0.0045 |
| IterativeImputer | 0.1315 ± 0.0091 | 0.1281 ± 0.0072 | 0.1298 ± 0.0066 | 0.1464 ± 0.0014 | 0.1486 ± 0.0010 |
| Mean/Mode (reference) | 0.1111 ± 0.0055 | 0.1108 ± 0.0028 | 0.1141 ± 0.0081 | 0.1332 ± 0.0029 | **0.1364** ± 0.0017 |

More results are in [`results/summary.md`](results/summary.md):

* numerical MAE and categorical error rate per method;
* the discrete-column ablation;
* a baseline ablation;
* the runs of the original code.

### 7.4 Comparison: what reproduces and what doesn't

**Reproduced**

* The TGAIN pipeline itself: the unmodified original code and the port give the same error for the same inputs (0.2987 vs 0.2995).
* The error range. **The original notebook's own printed result for the 30 % setting is 0.3179.** That matches our 0.3079 ± 0.031, not the 0.2605 reported in Table 5.2.
* The small gap between TGAIN with 3 models and with 1 model. The thesis itself calls the gain "rather small". In our runs 3 models help a little on categorical cells (error rate 0.44–0.48 vs 0.46–0.52) and are equal or slightly better on the corrected metric.

**Not reproduced**

* *TGAIN's absolute errors.* Ours are 0.02–0.05 higher at every rate. The thesis values sit at the lower edge of our seed distribution: 9 of 10 values lie below our mean, z between −0.6 and −2.2.
* *The ranking.* The thesis says TGAIN is best at every missing rate. In our runs **MissForest is clearly best at 10–70 %**. KNN and IterativeImputer are also better than TGAIN at low and medium rates. TGAIN (3 models) is worse than plain mean/mode imputation up to 50 %. Only at 90 % is TGAIN competitive: 1-model rank 2.0, roughly tied with the other methods. This partly agrees with the thesis claim that TGAIN gets relatively better at high missing rates.
* *The baseline values.* The thesis baselines, especially MissForest (0.31–0.40) and IterativeImputer (0.33–0.36), are much worse than ours (0.18–0.35 and 0.24–0.39).

## 8. Why the results differ

1. **Single run, small test set, high variance.** The test set has 50 rows (about 100–950 imputed cells), the masks were not seeded, and GAN training is stochastic. Between seeds, the same configuration varies by sd ≈ 0.02–0.03, with ranges up to 0.09. That is as large as most gaps in Table 5.2. One run per cell can't support the rankings in the thesis. The thesis numbers are consistently near the favourable end of our distribution, which suggests a lucky draw or run selection. The notebook's own 0.318 at 30 % supports this.
2. **Baseline configuration is unknown.** No code or settings are published for MissForest, MICE, KNN or IterativeImputer. Many details change their scores a lot:
   * one-hot vs label encoding;
   * rounding of category codes;
   * which rows the imputer is fitted on;
   * the R vs Python implementation.

   I tested one hypothesis: fitting the baselines only on the 50 incomplete test rows (`results/german_results_baselines_testonly.csv`). Most baseline errors rise by about 0.01–0.04, but MissForest stays best, so this alone doesn't explain the thesis baselines.
3. **Quirks in the original evaluation code**, kept in the "original metric" so the numbers stay comparable:
   * (a) the true and the imputed test sets are label-encoded and min-max scaled *separately*, so codes and scales can disagree between them;
   * (b) the score is `(num_err + α·cat_err)/2`, not the #numeric/#total and #categoric/#total weighting printed in §5.2.1;
   * (c) an off-by-one in `calc_avg_dissimilarity`;
   * (d) column types for the evaluation are detected on the 382 training rows, but TGAIN detects them on the 432-row input. So `Installment_rate_percentage`, `Present_residence_since` and `Number_existing_credits` are *categorical* during imputation but *numerical* during evaluation.

   The corrected metric (§7.3) fixes (a)–(c). The conclusions don't change.
4. **Design weaknesses of TGAIN that show up in the error breakdown:**
   * **Category codes are modelled as continuous values.** sdv 0.15 inferred them as float. Declaring them discrete in CTGAN (`--discrete-mode discrete`, 3 reps) only gives a small improvement at 30–90 % (e.g. 0.278 vs 0.294 at 50 %).
   * **The ranking distance mixes scales.** It uses the observed non-categorical columns, which include *categorical_unique* codes on a 0–9 scale (Purpose, Credit_history, Savings, …) next to numerical values in [0, 1]. Nearest-row selection is therefore driven by matching arbitrary category codes, not by the continuous variables.
   * **Conditioning is lost.** With ~10 categorical conditions, rejection sampling rarely matches all of them, and conditions are dropped at random. Many imputations end up close to unconditional samples, about as informative as the marginal distribution. That explains why TGAIN is near mean/mode level.
   * **The training set is tiny.** 382 rows means CTGAN does a single optimisation step per epoch (batch 500), so 300 epochs = 300 generator updates.
5. **Data reconstruction.** The original preprocessed CSV isn't public. My rebuild matches every published property exactly (§3), but small differences in label spelling or row order can't be ruled out. Label spelling doesn't affect the codes' meaning, only their alphabetical order.
6. **Software versions.** ctgan 0.12 / torch 2 vs ctgan 0.5 / torch 1.13 give different random streams, so results match only in distribution. The legacy run (§6) shows the distribution matches.
7. **A latent bug in the original pipeline at 10 %.** The output puts complete rows first and the notebook takes `.tail(50)`. A test row that gets *no* missing cell (likely at 10 %: P = 0.9²¹ ≈ 11 % per row) moves into the complete block, so predictions get compared with the wrong true rows. The port keeps the input order (`preserve_order=True`) to avoid this. `legacy/run_original_code.py` skips masks where this would happen.

## 9. Notes

* The upstream repository has no license. The algorithm and evaluation code here are re-implementations with attribution to O. Bannany (Utrecht University). The UCI German Credit data is CC BY 4.0.
* All raw per-run results are in `results/*.csv`. `python experiments/make_report.py` regenerates every table and figure.
