#!/usr/bin/env python3

"""
compare_case.py

Given a valid CASEID, compare:
  1. Actual 30-day postoperative BMI vs NN-predicted BMI
  2. Actual 30-day postoperative weight vs NN-predicted weight

Also prints the patient's phenotype score.

Usage:
    python compare_case.py 1922979
"""

import json
import importlib.util
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from model_utils import build_feature_matrix
import config
import scoring_engine


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "train_model", "05_train_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_case.py CASEID")
        sys.exit(1)

    case_id = sys.argv[1]

    # ------------------------------------------------------------
    # Load data and locked train/test split
    # ------------------------------------------------------------
    df = pd.read_parquet(
        "outputs/04_final_scored.parquet"
    ).reset_index(drop=True)

    with open("models/train_val_test_indices.json") as f:
        splits = json.load(f)

    train_idx = np.array(
        splits["train_indices"],
        dtype=int
    )

    test_idx = np.array(
        splits["test_indices"],
        dtype=int
    )

    # ------------------------------------------------------------
    # Find CASEID
    # ------------------------------------------------------------
    case_matches = df[
        df[config.CASEID_COLUMN].astype(str) == str(case_id)
    ]

    if len(case_matches) == 0:
        print(f"ERROR: CASEID {case_id} was not found.")
        sys.exit(1)

    if len(case_matches) > 1:
        print(
            f"ERROR: CASEID {case_id} appears "
            f"{len(case_matches)} times."
        )
        sys.exit(1)

    idx = int(case_matches.index[0])
    row = df.loc[idx]

    # ------------------------------------------------------------
    # Build the exact Step 5 feature matrix
    # ------------------------------------------------------------
    X, feature_to_variable, feature_to_domain, _ = (
        build_feature_matrix(df)
    )

    # Train-only imputation
    imputation_medians = {}

    for col in X.columns:
        if "__SEVERITY" not in col:
            continue

        train_values = X.loc[train_idx, col]
        median_value = train_values.median()

        if pd.isna(median_value):
            median_value = 0.0

        imputation_medians[col] = float(median_value)

    for col, median_value in imputation_medians.items():
        X[col] = X[col].fillna(median_value)

    X = X.fillna(0.0)

    # Exact scaler procedure
    scaler = StandardScaler()
    scaler.fit(X.loc[train_idx])

    X_scaled = scaler.transform(X)

    # ------------------------------------------------------------
    # Load saved neural networks
    # ------------------------------------------------------------
    train_model = load_training_module()

    device = torch.device("cpu")

    bmi_model = train_model.BMI_Model(
        input_features=X_scaled.shape[1]
    )
    bmi_model.load_state_dict(
        torch.load(
            "models/mlp_bmi_30d.pth",
            map_location=device
        )
    )
    bmi_model.eval()

    weight_model = train_model.BMI_Model(
        input_features=X_scaled.shape[1]
    )
    weight_model.load_state_dict(
        torch.load(
            "models/mlp_weight_30d.pth",
            map_location=device
        )
    )
    weight_model.eval()

    patient_x = torch.tensor(
        X_scaled[idx:idx + 1],
        dtype=torch.float32
    )

    with torch.no_grad():
        predicted_bmi = float(
            bmi_model(patient_x).item()
        )

        predicted_weight = float(
            weight_model(patient_x).item()
        )

    # ------------------------------------------------------------
    # Actual outcomes
    # ------------------------------------------------------------
    actual_bmi = float(
        row["OUTCOME_BMI_30D"]
    )

    actual_weight = float(
        row["OUTCOME_WEIGHT_30D_KG"]
    )

    # ------------------------------------------------------------
    # Phenotype score using the project's scoring engine
    # ------------------------------------------------------------
    clinical_weights = (
        scoring_engine.default_variable_weights()
    )

    effective_weights = (
        scoring_engine.apply_flag_count_weight_nudge(
            df,
            clinical_weights
        )
    )

    domain_weights = (
        scoring_engine.default_domain_weights()
    )

    domain_scores, total_score = (
        scoring_engine.full_scoring_pipeline(
            df,
            effective_weights,
            domain_weights
        )
    )

    phenotype_score = float(
        total_score.loc[idx]
    )

    # ------------------------------------------------------------
    # Determine whether this patient is in the test set
    # ------------------------------------------------------------
    is_test = idx in set(test_idx)

    # ------------------------------------------------------------
    # Print comparison
    # ------------------------------------------------------------
    print()
    print("=" * 65)
    print("CASE COMPARISON")
    print("=" * 65)

    print(f"CASEID:               {case_id}")
    print(f"Dataset row:          {idx}")
    print(f"Locked test set:      {'YES' if is_test else 'NO'}")

    print()
    print("30-DAY BMI")
    print("-" * 65)
    print(f"Actual BMI:           {actual_bmi:.3f}")
    print(f"Predicted BMI:        {predicted_bmi:.3f}")
    print(f"Absolute error:       {abs(predicted_bmi - actual_bmi):.3f}")
    print(f"Prediction error:     {predicted_bmi - actual_bmi:+.3f}")

    print()
    print("30-DAY WEIGHT")
    print("-" * 65)
    print(f"Actual weight:        {actual_weight:.3f} kg")
    print(f"Predicted weight:     {predicted_weight:.3f} kg")
    print(
        f"Absolute error:       "
        f"{abs(predicted_weight - actual_weight):.3f} kg"
    )
    print(
        f"Prediction error:     "
        f"{predicted_weight - actual_weight:+.3f} kg"
    )

    print()
    print("PHENOTYPE SCORE")
    print("-" * 65)

    for domain in domain_weights:
        score = float(
            domain_scores.loc[
                idx,
                f"{domain}_SCORE"
            ]
        )
        print(f"{domain:25s} {score:.3f}")

    print("-" * 65)
    print(f"{'TOTAL':25s} {phenotype_score:.3f}")

    print("=" * 65)


if __name__ == "__main__":
    main()
