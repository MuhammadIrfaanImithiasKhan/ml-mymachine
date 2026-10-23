#!/usr/bin/env python3

import argparse
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Generated results that can be archived before a new run.
# This is an initial list, not yet a complete pipeline reset.
RESET_FILES = [
    "outputs/04_final_scored_v2.parquet",
    "outputs/optuna_history_v2.png",
    "outputs/09_xgboost_test_predictions_BMI_v2.csv",
    "outputs/09_xgboost_test_predictions_WEIGHT_v2.csv",
    "outputs/09_xgboost_actual_vs_predicted_BMI_v2.png",
    "outputs/09_xgboost_actual_vs_predicted_WEIGHT_v2.png",
    "outputs/10_paired_model_comparison_v2.json",
    "outputs/10_paired_test_predictions_BMI_30D_v2.csv",
    "outputs/10_paired_test_predictions_WEIGHT_30D_KG_v2.csv",
    "outputs/training_leakage_audit_v2.json",
    "outputs/training_leakage_audit_v2.txt",
    "models/training_metrics_v2.json",
    "models/model_comparison_metrics_v2.json",
    "models/xgboost_metrics_v2.json",
    "models/feature_metadata_v2.json",
    "models/feature_scaler_v2.joblib",
    "models/mlp_bmi_30d_v2.pth",
    "models/mlp_weight_30d_v2.pth",
]

# These files must never be removed by this utility.
PROTECTED_FILES = [
    "outputs/01_raw_extract.parquet",
    "outputs/02_master_flagged.parquet",
    "outputs/03_scored.parquet",
    "models/train_val_test_indices.json",
    "models/train_val_test_indices_v2.json",
    "models/learned_weights_v2.json",
    "models/clinical_baseline_weights.json",
]

def reset(apply):
    print("\nV2 PIPELINE RESET")
    print("=" * 65)

    existing = []

    for relative in RESET_FILES:
        path = ROOT / relative
        if path.is_file():
            existing.append((relative, path))

    if not existing:
        print("No listed generated files found.")
        return

    print("\nFiles selected for archiving:\n")

    for relative, path in existing:
        print(f"  {relative} ({path.stat().st_size:,} bytes)")

    print(f"\nTotal files: {len(existing)}")

    if not apply:
        print("\nDRY RUN ONLY: No files changed.")
        print("To archive these files, run:")
        print("  python pipeline_manager.py reset --apply")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive = ROOT / "archives" / f"pipeline_reset_{timestamp}"

    # Copy and verify all files before removing any originals.
    for relative, source in existing:
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        if (
            source.stat().st_size != destination.stat().st_size
            or source.read_bytes() != destination.read_bytes()
        ):
            raise RuntimeError(
                f"Backup verification failed: {relative}"
            )

    print(f"\nVerified archive: {archive.relative_to(ROOT)}")

    for relative, source in existing:
        source.unlink()
        print(f"Archived: {relative}")

    print("\nRESET COMPLETE")
    print("Protected inputs and splits were not modified.")



def score_fixed(weights_path, apply):
    """Regenerate v2 phenotype scores without running Optuna."""
    import json
    import numpy as np
    import pandas as pd
    import scoring_engine

    source = ROOT / "outputs/03_scored.parquet"
    destination = ROOT / "outputs/04_final_scored_v2.parquet"
    weights_file = Path(weights_path).expanduser()

    if not weights_file.is_absolute():
        weights_file = ROOT / weights_file

    if not source.is_file():
        raise FileNotFoundError(source)

    if not weights_file.is_file():
        raise FileNotFoundError(weights_file)

    if weights_file.resolve() == destination.resolve():
        raise ValueError("Weights path cannot be the output dataset.")

    with weights_file.open() as f:
        weights = json.load(f)

    variable_weights = weights["variable_weights"]
    domain_weights = weights["domain_weights"]

    print("\\nFIXED-WEIGHT SCORING")
    print("=" * 65)
    print(f"Input:   {source.relative_to(ROOT)}")
    print(f"Weights: {weights_file}")
    print(f"Output:  {destination.relative_to(ROOT)}")
    print("Optuna:  SKIPPED")

    for domain, values in variable_weights.items():
        if not isinstance(values, dict) or not values:
            raise ValueError(f"Invalid weights for {domain}")

        numbers = np.asarray(list(values.values()), dtype=float)

        if not np.isfinite(numbers).all() or (numbers < 0).any():
            raise ValueError(f"Invalid numeric weights for {domain}")

        if not np.isclose(numbers.sum(), 100.0, atol=1e-4):
            raise ValueError(f"Weights do not sum to 100: {domain}")

    domain_numbers = np.asarray(list(domain_weights.values()), dtype=float)

    if not np.isfinite(domain_numbers).all():
        raise ValueError("Domain weights contain nonfinite values")

    if (domain_numbers < 0).any():
        raise ValueError("Domain weights contain negative values")

    if not np.isclose(domain_numbers.sum(), 100.0, atol=1e-4):
        raise ValueError("Domain weights do not sum to 100")

    if not apply:
        print("\\nDRY RUN ONLY: No files changed.")
        print("To generate scores:")
        print(
            "  python pipeline_manager.py score-fixed "
            "--weights models/learned_weights_v2.json --apply"
        )
        return

    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists. "
            "Archive it using reset before regenerating."
        )

    df = pd.read_parquet(source)

    domain_scores, total_score = (
        scoring_engine.full_scoring_pipeline(
            df, variable_weights, domain_weights
        )
    )

    final_df = df.copy()

    for column in domain_scores.columns:
        final_df[column] = domain_scores[column]

    final_df["TOTAL_PHENOTYPE_SCORE"] = total_score

    temporary = destination.with_name(
        destination.stem + ".temporary.parquet"
    )

    if temporary.exists():
        raise FileExistsError(temporary)

    try:
        final_df.to_parquet(temporary, index=True)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"\\nSUCCESS: Saved {len(final_df):,} scored patients.")
    print(f"Output: {destination.relative_to(ROOT)}")

def main():
    parser = argparse.ArgumentParser(
        description="Manage obesity digital-twin v2 pipeline outputs."
    )

    subparsers = parser.add_subparsers(dest="command")

    reset_parser = subparsers.add_parser(
        "reset",
        help="Preview or archive generated v2 outputs."
    )

    reset_parser.add_argument(
        "--apply",
        action="store_true",
        help="Archive verified copies and remove active originals."
    )

    fixed_parser = subparsers.add_parser(
        "score-fixed",
        help="Regenerate v2 phenotype scores with fixed weights."
    )

    fixed_parser.add_argument(
        "--weights",
        required=True,
        help="Path to fixed variable/domain weights JSON."
    )

    fixed_parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the scored dataset."
    )

    args = parser.parse_args()

    if args.command == "reset":
        reset(args.apply)
    elif args.command == "score-fixed":
        score_fixed(args.weights, args.apply)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
