#!/usr/bin/env python3

import argparse
import pandas as pd


ADIPOSITY_WEIGHTS = {
    "BMI_HIGH_BAR": 29.93439550140582,
    "BMI": 25.248359887535145,
    "WGT_HIGH_BAR": 20.56232427366448,
    "WGT_CLOSEST": 11.19025304592315,
    "VENOUS_STASIS": 5.567010309278351,
}


def calculate_adiposity(row):
    """Manually reconstruct the Adiposity score for one patient."""
    weighted_sum = 0.0
    weight_sum = 0.0

    severity_columns = {
        "BMI_HIGH_BAR": "BMI_HIGH_BAR__SEVERITY",
        "BMI": "BMI__SEVERITY",
        "WGT_HIGH_BAR": "WGT_HIGH_BAR__SEVERITY",
        "WGT_CLOSEST": "WGT_CLOSEST__SEVERITY",
        "VENOUS_STASIS": "VENOUS_STASIS__SEVERITY",
    }

    for variable, weight in ADIPOSITY_WEIGHTS.items():
        value = row[severity_columns[variable]]

        # Match the scoring pipeline: missing values do not contribute
        # to either the numerator or denominator.
        if pd.notna(value):
            weighted_sum += value * weight
            weight_sum += weight

    return weighted_sum / weight_sum if weight_sum else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Manually validate a patient's Adiposity phenotype score using CASEID."
    )
    parser.add_argument(
        "caseid",
        help="CASEID of the patient to validate.",
    )
    parser.add_argument(
        "--file",
        default="outputs/03_scored.parquet",
        help="Path to the scored parquet file.",
    )

    args = parser.parse_args()

    df = pd.read_parquet(args.file)

    if "CASEID" not in df.columns:
        raise SystemExit("ERROR: CASEID column was not found in the parquet file.")

    # Compare as strings so numeric/string CASEIDs both work.
    matches = df[df["CASEID"].astype(str) == str(args.caseid)]

    if len(matches) == 0:
        raise SystemExit(f"ERROR: CASEID {args.caseid} was not found.")

    if len(matches) > 1:
        raise SystemExit(
            f"ERROR: CASEID {args.caseid} appears {len(matches)} times. "
            "Expected a unique CASEID."
        )

    row = matches.iloc[0]

    manual_score = calculate_adiposity(row)
    stored_score = row["Adiposity_SCORE"]
    difference = abs(stored_score - manual_score)

    print("=" * 60)
    print("PHENOTYPE SCORE VALIDATION")
    print("=" * 60)
    print(f"CASEID:             {args.caseid}")

    print()
    print("Stored Adiposity:   {:.9f}".format(stored_score))
    print("Manual Adiposity:   {:.9f}".format(manual_score))
    print("Difference:          {:.12f}".format(difference))

    print()
    if difference < 1e-6:
        print("RESULT: PASS")
        print("The manual calculation matches the pipeline score.")
    else:
        print("RESULT: FAIL")
        print("The manual calculation does not match the pipeline score.")

    print("=" * 60)


if __name__ == "__main__":
    main()
