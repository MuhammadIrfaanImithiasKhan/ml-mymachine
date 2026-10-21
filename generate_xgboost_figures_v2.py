import os
import json
import importlib.machinery

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Reuse the exact data-preparation code from the XGBoost experiment.
experiment = importlib.machinery.SourceFileLoader(
    "xgb_experiment", "09_xgboost_models_v2.py"
).load_module()

(
    X_scaled, y_bmi, y_weight,
    train_idx, val_idx, test_idx
) = experiment.prepare_data()

with open("models/xgboost_metrics_v2.json", "r") as f:
    saved_metrics = json.load(f)

os.makedirs("outputs", exist_ok=True)


def make_figure(target, y, filename, title, units):
    valid = y.notna()

    train_rows = [i for i in train_idx if valid.loc[i]]
    val_rows = [i for i in val_idx if valid.loc[i]]
    test_rows = [i for i in test_idx if valid.loc[i]]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        early_stopping_rounds=50,
        tree_method="hist",
        n_jobs=4,
        random_state=42
    )

    model.fit(
        X_scaled.loc[train_rows], y.loc[train_rows],
        eval_set=[(X_scaled.loc[val_rows], y.loc[val_rows])],
        verbose=False
    )

    observed = y.loc[test_rows].to_numpy()
    predicted = model.predict(X_scaled.loc[test_rows])

    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    mae = float(mean_absolute_error(observed, predicted))
    r2 = float(r2_score(observed, predicted))

    expected = saved_metrics[target]["test_metrics"]

    # Do not save a figure if the recreated results disagree
    # materially with the recorded experiment.
    for name, actual in [("RMSE", rmse), ("MAE", mae), ("R2", r2)]:
        if abs(actual - expected[name]) > 0.02:
            raise ValueError(
                f"{target} {name} mismatch: recreated={actual:.4f}, "
                f"saved={expected[name]:.4f}. Figure not saved."
            )

    print(
        f"{target}: RMSE={rmse:.3f}, MAE={mae:.3f}, "
        f"R²={r2:.3f}, n={len(observed):,}"
    )


    # Export all test predictions and print a compact 25-row preview.
    import pandas as pd

    prediction_table = pd.DataFrame({
        "X_Actual": observed,
        "Y_Predicted": predicted,
        "Error_Predicted_Minus_Actual": predicted - observed
    })

    target_suffix = "BMI" if "BMI" in target else "WEIGHT"
    csv_path = (
        f"outputs/09_xgboost_test_predictions_{target_suffix}_v2.csv"
    )
    prediction_table.to_csv(csv_path, index=False)

    print(f"\n{target} — first 25 actual vs. predicted values")
    print(prediction_table.head(25).round(3).to_string(index=False))
    print(f"Full table saved to: {csv_path}")
    print(f"Actual range:    {observed.min():.3f} to {observed.max():.3f}")
    print(f"Predicted range: {predicted.min():.3f} to {predicted.max():.3f}")


    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        observed, predicted,
        s=5, alpha=0.15, rasterized=True
    )

    lower = min(observed.min(), predicted.min())
    upper = max(observed.max(), predicted.max())
    ax.plot(
        [lower, upper], [lower, upper],
        linestyle="--", linewidth=2, label="Perfect prediction"
    )

    ax.set_xlabel(f"Observed {title} ({units})")
    ax.set_ylabel(f"Predicted {title} ({units})")
    ax.set_title(f"XGBoost: Observed vs. Predicted {title}")
    ax.text(
        0.05, 0.95,
        f"Test observations: {len(observed):,}\n"
        f"RMSE = {rmse:.3f}\nMAE = {mae:.3f}\nR² = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {filename}")


make_figure(
    "BMI_30D", y_bmi,
    "outputs/09_xgboost_actual_vs_predicted_BMI_v2.png",
    "BMI at Follow-up", "kg/m²"
)

make_figure(
    "WEIGHT_30D_KG", y_weight,
    "outputs/09_xgboost_actual_vs_predicted_WEIGHT_v2.png",
    "Weight at Follow-up", "kg"
)

print("Both XGBoost figures generated and checked.")
