"""Main experiment of the thesis: Table 5.2 / Fig. 5.4 - prediction error on the
German Credit data for missing rates 10..90 %, TGAIN (1 and 3 models) vs.
MissForest, MICE, KNNImputer and IterativeImputer.

Protocol (identical to TGAIN_illustration.ipynb):
  * keep the 432 complete rows, shuffle with random_state=0,
  * first 50 rows = test set, remaining 382 = training set,
  * remove each test cell independently with probability = missing rate (MCAR),
  * impute train+test together, evaluate on the 50 test rows.

Results are appended row by row to a CSV, so the script can be interrupted and
resumed. Example:
    python experiments/run_german.py --reps 5 --out results/german_results.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tgain import (data_imputation_completer, impute_baseline,  # noqa: E402
                   evaluate_imputations_original, evaluate_imputations_corrected)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASELINES = ["MissForest", "MICE", "KNNImputer", "IterativeImputer"]


def make_split(data, n_test=50, split_seed=0):
    complete = shuffle(data.dropna(), random_state=split_seed)
    return complete[n_test:], complete[:n_test]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "german_data_preprocessed.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "german_results.csv"))
    ap.add_argument("--rates", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--g-lr", type=float, default=2e-4)
    ap.add_argument("--d-lr", type=float, default=2e-4)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--discrete-mode", default="original", choices=["original", "discrete"])
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-tgain", action="store_true")
    ap.add_argument("--baselines", default=",".join(BASELINES),
                    help="comma-separated: MissForest,MICE,KNNImputer,IterativeImputer,MeanMode")
    ap.add_argument("--baseline-fit", default="all", choices=["all", "test"],
                    help="fit baselines on train+test (default) or only on the 50 test rows")
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)

    data = pd.read_csv(args.data, keep_default_na=False, na_values=[''])
    train, test = make_split(data, args.n_test)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(zip(prev.method, prev.rate.round(2), prev.rep))

    def record(method, rate, rep, test_imputed, mask, secs):
        s_o, p_o = evaluate_imputations_original(train, test, mask, test_imputed,
                                                 args.threshold, return_parts=True)
        s_c, p_c = evaluate_imputations_corrected(data, test, mask, test_imputed,
                                                  args.threshold, return_parts=True)
        row = dict(method=method, rate=rate, rep=rep, error_original=s_o,
                   error_corrected=s_c, num_err=p_c["num_err"], cat_err=p_c["cat_err"],
                   alpha_original=p_o["alpha"], alpha_corrected=p_c["alpha"],
                   n_missing=int(mask.sum()), seconds=round(secs, 1),
                   discrete_mode=args.discrete_mode, epochs=args.epochs)
        pd.DataFrame([row]).to_csv(args.out, mode="a", index=False,
                                   header=not os.path.exists(args.out))
        print(f"  {method:<18} rate={rate:.1f} rep={rep} "
              f"err_orig={s_o:.4f} err_corr={s_c:.4f} ({secs:.0f}s)", flush=True)

    for rep in range(args.rep_start, args.rep_start + args.reps):
        for rate in [float(r) for r in args.rates.split(",")]:
            mask = np.random.RandomState(int(rate * 100) * 1000 + rep).random(test.shape) < rate
            test_na = test.mask(mask)
            total = pd.concat([train, test_na], ignore_index=True, sort=False)
            n = args.n_test
            print(f"rate={rate} rep={rep} missing cells={mask.sum()}", flush=True)

            if not args.no_tgain and ("TGAIN (3 models)", round(rate, 2), rep) not in done:
                t0 = time.time()
                res, det = data_imputation_completer(
                    total, nr_epochs=args.epochs, distinct_ratio_threshold=args.threshold,
                    generator_lr=args.g_lr, discriminator_lr=args.d_lr,
                    discrete_mode=args.discrete_mode, seed=rep * 100 + int(rate * 100),
                    return_details=True, preserve_order=True)
                secs = time.time() - t0
                record("TGAIN (1 model)", rate, rep,
                       det["per_model"][0].tail(n).reset_index(drop=True), mask, secs / 3)
                record("TGAIN (3 models)", rate, rep, res.tail(n).reset_index(drop=True),
                       mask, secs)
            if not args.no_baselines:
                fit_rows = None
                if args.baseline_fit == "test":
                    fit_rows = np.r_[np.zeros(len(train), bool), np.ones(n, bool)]
                for b in args.baselines.split(","):
                    if (b, round(rate, 2), rep) in done:
                        continue
                    t0 = time.time()
                    res = impute_baseline(b, total, args.threshold, seed=rep,
                                          fit_rows=fit_rows)
                    record(b, rate, rep, res.tail(n).reset_index(drop=True), mask,
                           time.time() - t0)


if __name__ == "__main__":
    main()
