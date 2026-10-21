import os
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model_utils import build_feature_matrix
from importlib.machinery import SourceFileLoader


# ------------------------------------------------------------
# Load the existing model class WITHOUT running training
# ------------------------------------------------------------

train_module = SourceFileLoader(
    "train_module",
    "05_train_model.py"
).load_module()

BMI_Model = train_module.BMI_Model

DEVICE = torch.device("cpu")

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print("Loading data...")

df = pd.read_parquet(
    os.path.join(
        OUTPUT_DIR,
        "04_final_scored.parquet"
    )
)

print(f"Loaded {len(df):,} rows.")


# ------------------------------------------------------------
# Build feature matrix exactly like 05_train_model.py
# ------------------------------------------------------------

X_initial, feature_to_variable, feature_to_domain, _ = (
    build_feature_matrix(df)
)

feature_columns = list(X_initial.columns)

X = X_initial.copy()

for col in feature_columns:

    if col.endswith("__SEVERITY"):

        variable_name = feature_to_variable[col]

        if variable_name in df.columns:

            source_col = f"{variable_name}__SEVERITY"

            if source_col in df.columns:
                X[col] = df[source_col]


# ------------------------------------------------------------
# Load permanent test split
# ------------------------------------------------------------

with open(
    os.path.join(
        MODEL_DIR,
        "train_val_test_indices.json"
    ),
    "r"
) as f:

    split_data = json.load(f)


test_idx = np.array(
    split_data["test_indices"],
    dtype=int
)


# ------------------------------------------------------------
# Reproduce train-safe preprocessing
# ------------------------------------------------------------

with open(
    os.path.join(
        MODEL_DIR,
        "feature_metadata.json"
    ),
    "r"
) as f:

    metadata = json.load(f)


imputation_medians = metadata["imputation_medians"]
normalization_stats = metadata["normalization_stats"]


# HGT and OPYEAR normalization

for variable_name in ["HGT", "OPYEAR"]:

    source_col = f"{variable_name}__STD"
    norm_col = f"{variable_name}__NORM"

    train_mean = normalization_stats[variable_name]["mean"]
    train_std = normalization_stats[variable_name]["std"]

    if train_std == 0:

        X[norm_col] = 0.0

    else:

        X[norm_col] = (
            pd.to_numeric(
                df[source_col],
                errors="coerce"
            ) - train_mean
        ) / train_std


# Severity imputation

for col, median_value in imputation_medians.items():

    X[col] = X[col].fillna(
        median_value
    )


X = X.fillna(0.0)


# ------------------------------------------------------------
# Load the EXACT scaler used during training
# ------------------------------------------------------------

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_scaler.joblib"
    )
)

X_scaled = pd.DataFrame(
    scaler.transform(X),
    index=X.index,
    columns=X.columns
)


# ------------------------------------------------------------
# Helper for plotting
# ------------------------------------------------------------

def make_plot(
    target_name,
    model_filename,
    outcome_column,
    test_indices,
    output_filename,
    units
):

    print()
    print(f"Generating {target_name} figure...")

    # Match 05_train_model.py exactly:
    # only evaluate patients with a valid target outcome.
    valid_outcomes = pd.to_numeric(
        df.loc[
            test_indices,
            outcome_column
        ],
        errors="coerce"
    ).notna()

    valid_test_indices = np.array(
        test_indices
    )[valid_outcomes.to_numpy()]

    X_test = X_scaled.loc[
        valid_test_indices
    ].to_numpy(
        dtype=np.float32
    )

    y_test = pd.to_numeric(
        df.loc[
            valid_test_indices,
            outcome_column
        ],
        errors="coerce"
    ).to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Load saved model
    # --------------------------------------------------------

    model = BMI_Model(
        input_features=X_test.shape[1]
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            os.path.join(
                MODEL_DIR,
                model_filename
            ),
            map_location=DEVICE
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    with torch.no_grad():

        predictions = (
            model(
                torch.tensor(
                    X_test,
                    dtype=torch.float32
                )
            )
            .cpu()
            .numpy()
            .reshape(-1)
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"{target_name}: "
        f"RMSE={rmse:.3f}, "
        f"MAE={mae:.3f}, "
        f"R²={r2:.3f}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        y_test,
        predictions,
        s=5,
        alpha=0.15
    )

    minimum = min(
        y_test.min(),
        predictions.min()
    )

    maximum = max(
        y_test.max(),
        predictions.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linewidth=2
    )

    plt.xlabel(
        f"Observed {target_name} ({units})"
    )

    plt.ylabel(
        f"Predicted {target_name} ({units})"
    )

    plt.title(
        f"{target_name}: Observed vs Predicted"
    )

    plt.text(
        0.05,
        0.95,
        f"RMSE = {rmse:.3f}\n"
        f"MAE = {mae:.3f}\n"
        f"R² = {r2:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8
        )
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved {output_path}"
    )


# ------------------------------------------------------------
# Generate BMI figure
# ------------------------------------------------------------

make_plot(
    target_name="BMI 30-Day Outcome",
    model_filename="mlp_bmi_30d.pth",
    outcome_column="OUTCOME_BMI_30D",
    test_indices=test_idx,
    output_filename="05_actual_vs_predicted_BMI.png",
    units="kg/m²"
)


# ------------------------------------------------------------
# Generate weight figure
# ------------------------------------------------------------

make_plot(
    target_name="Weight 30-Day Outcome",
    model_filename="mlp_weight_30d.pth",
    outcome_column="OUTCOME_WEIGHT_30D_KG",
    test_indices=test_idx,
    output_filename="05_actual_vs_predicted_WEIGHT.png",
    units="kg"
)


print()
print("Model figures generated successfully.")
