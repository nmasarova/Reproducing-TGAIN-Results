"""Rebuild the `german_data_preprocessed.csv` used by the original TGAIN notebook.

The original notebook loads `rawData/german/german_data_preprocessed.csv`, which is
not included in the public repository. This script reconstructs it from the UCI
Statlog (German Credit) data so that it matches every property reported in the
thesis (Table 5.1, Fig. 5.1) and the `.head()` output printed in the notebook:

* 999 rows  -> the original file was read without `header=None`, so the first
               UCI record became the header and was lost.
* 21 columns (20 attributes + target), 3 continuous + 18 categorical.
* Exactly three columns contain missing values, giving 432 complete rows and
  567 incomplete rows (8 missingness patterns, Fig. 5.1):
    - Status_existing_account: A14 ("no checking account")     -> NaN
    - Savings:                 A65 ("unknown / no savings")    -> NaN
    - Property:                A124 ("unknown / no property")  -> NaN
* Human-readable category labels as seen in the notebook output
  (e.g. '0<=x<200', 'credits_paid_till_now', 'male_single').

Usage:
    python data/prepare_german.py            # downloads from UCI
    python data/prepare_german.py --raw path/to/german.data
"""
import argparse
import os
import urllib.request

import pandas as pd

UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "statlog/german/german.data")

COLUMNS = [
    "Status_existing_account", "Month", "Credit_history", "Purpose",
    "Credit_amount", "Savings", "Employment_since_years",
    "Installment_rate_percentage", "Gender_and_relationship_status",
    "Other_debtors", "Present_residence_since", "Property", "Age",
    "Other_installment_plans", "Housing", "Number_existing_credits", "Job",
    "Number_people_liable", "Telephone", "Foreign_worker", "Credit_risk",
]

NA = None  # marker for values that become missing

MAPS = {
    "Status_existing_account": {"A11": "x<0", "A12": "0<=x<200", "A13": "x>200", "A14": NA},
    # NB: the notebook shows the label 'None' for A30. With pandas >= 2 the string
    # "None" is parsed as NaN by default, so we use an unambiguous label instead.
    "Credit_history": {"A30": "no_credits_taken", "A31": "all_credits_paid_duly",
                       "A32": "credits_paid_till_now", "A33": "delay_in_earlier_payments",
                       "A34": "critical_account"},
    "Purpose": {"A40": "car_new", "A41": "car_used", "A42": "furniture/equipment",
                "A43": "radio/television", "A44": "domestic_appliances", "A45": "repairs",
                "A46": "education", "A47": "vacation", "A48": "retraining",
                "A49": "business", "A410": "others"},
    "Savings": {"A61": "x<100", "A62": "100<=x<500", "A63": "500<=x<1000",
                "A64": "x>=1000", "A65": NA},
    "Employment_since_years": {"A71": "unemployed", "A72": "x<1", "A73": "1<=x<4",
                               "A74": "4<=x<7", "A75": "x>=7"},
    "Gender_and_relationship_status": {"A91": "male_divorced", "A92": "female_divorced",
                                       "A93": "male_single", "A94": "male_married",
                                       "A95": "female_single"},
    "Other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "Property": {"A121": "real_estate", "A122": "building_society_savings",
                 "A123": "car_or_other", "A124": NA},
    "Other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "Housing": {"A151": "rent", "A152": "own", "A153": "for_free"},
    "Job": {"A171": "unskilled_non_resident", "A172": "unskilled_resident",
            "A173": "skilled", "A174": "highly_qualified"},
    "Telephone": {"A191": "none", "A192": "yes"},
    "Foreign_worker": {"A201": "yes", "A202": "no"},
}


def build(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, sep=" ", header=None, names=COLUMNS)
    df = df.iloc[1:].reset_index(drop=True)  # reproduce the lost first record
    for col, mapping in MAPS.items():
        df[col] = df[col].map(mapping).astype(object)
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(here, "german.data"))
    ap.add_argument("--out", default=os.path.join(here, "german_data_preprocessed.csv"))
    args = ap.parse_args()
    if not os.path.exists(args.raw):
        print(f"Downloading {UCI_URL}")
        urllib.request.urlretrieve(UCI_URL, args.raw)
    df = build(args.raw)
    df.to_csv(args.out, index=False)
    n_complete = len(df.dropna())
    print(f"rows={len(df)} cols={df.shape[1]} complete={n_complete} "
          f"incomplete={len(df) - n_complete}")
    pats = df.isna().value_counts(normalize=True).round(3)
    print("missingness pattern proportions (cf. thesis Fig. 5.1):")
    print(pats.to_string())


if __name__ == "__main__":
    main()
