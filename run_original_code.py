"""Run the *unmodified* original TGAIN code (tabularGAIN.py + UtilFunctions.py
from github.com/hqahtan/GANS_For_Data_Imputation) in a legacy environment
(Python 3.8, sdv 0.15.0 - see requirements-legacy.txt) on the same split /
masks as experiments/run_german.py. Used to cross-check the modern port.

    git clone https://github.com/hqahtan/GANS_For_Data_Imputation original_code
    python legacy/run_original_code.py --code original_code --rates 0.3 --reps 1

The original pipeline outputs complete rows first, so (like the notebook) we take
`.tail(50)`. To avoid the mis-alignment that happens when a test row receives no
missing value, masks where a test row is fully observed are skipped
(re-drawn with the next seed) - this matters only at the 10 % rate.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="folder with the original .py files")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "german_data_preprocessed.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "german_results_legacy.csv"))
    ap.add_argument("--rates", default="0.3")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.code))
    from tabularGAIN import data_imputation_completer
    from UtilFunctions import evaluate_imputations

    # keep_default_na=False + na_values=[''] : identical parsing under pandas 1.x/2.x
    data = pd.read_csv(args.data, keep_default_na=False, na_values=[""])
    complete = shuffle(data.dropna(), random_state=0)
    test, train = complete[:50], complete[50:]
    for rep in range(args.reps):
        for rate in [float(r) for r in args.rates.split(",")]:
            mask = np.random.RandomState(int(rate * 100) * 1000 + rep).random(test.shape) < rate
            if (~mask.any(axis=1)).any():
                print("skip: a test row has no missing value under this mask")
                continue
            total = pd.concat([train, test.mask(mask)], ignore_index=True, sort=False)
            t0 = time.time()
            res = data_imputation_completer(total, nr_epochs=args.epochs,
                                            distinct_ratio_threshold=0.01,
                                            generator_lr=0.0002, discriminator_lr=0.0002,
                                            verbose=False)
            imp = res.tail(50)
            err = evaluate_imputations(train, test, mask, imp, 0.01)
            row = dict(method="TGAIN (3 models) - original code", rate=rate, rep=rep,
                       error_original=err, seconds=round(time.time() - t0, 1))
            pd.DataFrame([row]).to_csv(args.out, mode="a", index=False,
                                       header=not os.path.exists(args.out))
            imp.to_csv(args.out.replace(".csv", f"_imputed_r{int(rate*100)}_{rep}.csv"),
                       index=False)
            print(row, flush=True)


if __name__ == "__main__":
    main()
