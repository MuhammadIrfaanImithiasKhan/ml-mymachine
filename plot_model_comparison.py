import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(".")
OUT = ROOT / "outputs" / "model_comparison_graphs"

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main():
    mlp_data = load_json("models/training_metrics_v2.json")
    rf_data = load_json("models/model_comparison_metrics_v2.json")
    xgb_data = load_json("models/xgboost_metrics_v2.json")

    targets = [
        ("BMI_30D", "BMI at 30 Days"),
        ("WEIGHT_30D_KG", "Weight at 30 Days (kg)"),
    ]

    models = ["MLP", "Random Forest", "XGBoost"]
    metrics = [
        ("RMSE", "RMSE (lower is better)", "rmse"),
        ("MAE", "MAE (lower is better)", "mae"),
        ("R2", "R² (higher is better)", "r2"),
    ]

    # Collect test-set metrics from each model's saved results.
    results = {}
    for target, _ in targets:
        results[target] = {
            "MLP": mlp_data["TEST_SET"][target],
            "Random Forest": rf_data[target]["random_forest"]["test"],
            "XGBoost": xgb_data[target]["test_metrics"],
        }

    # Guard against missing metrics before creating any graphs.
    for target, _ in targets:
        for model in models:
            for metric, _, _ in metrics:
                if metric not in results[target][model]:
                    raise ValueError(
                        f"Missing {metric} for {target}, {model}"
                    )
                value = results[target][model][metric]
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(
                        f"Invalid {metric} for {target}, {model}: {value}"
                    )

    OUT.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, _ in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Three-Model Test-Set Comparison: {metric}",
            fontsize=14
        )

        for ax, (target, target_label) in zip(axes, targets):
            values = [
                results[target][model][metric] for model in models
            ]
            bars = ax.bar(models, values, width=0.65)
            ax.set_title(target_label)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", rotation=12)

            for bar, value in zip(bars, values):
                ax.annotate(
                    f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9
                )

        fig.tight_layout()
        filename = f"{metric.lower()}_three_model_comparison.png"
        path = OUT / filename
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved image: {path.resolve()}")

    print("\nAll 3 comparison graphs generated successfully.")
    print(f"Output folder: {OUT.resolve()}")

if __name__ == "__main__":
    main()
