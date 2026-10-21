import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

input_path = Path("data/MBSAQIP_PUF_MAIN_2023.xlsx")
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(
    input_path,
    sheet_name="MBSAQIP_PUF_MAIN",
    usecols=[
        "DTBMI_30D",
        "BMI_CLOSEST30D",
        "FOLLOW_30DAYS_BAR",
        "REOP30",
    ],
)

# Summarize the recorded timing of BMI measurements.
days = df["DTBMI_30D"].dropna()

summary = pd.DataFrame([{
    "total_patients": len(df),
    "patients_with_timing": len(days),
    "missing_timing": df["DTBMI_30D"].isna().sum(),
    "missing_timing_pct": round(
        df["DTBMI_30D"].isna().mean() * 100, 2
    ),
    "median_day": days.median(),
    "mean_day": round(days.mean(), 2),
    "minimum_day": days.min(),
    "maximum_day": days.max(),
    "recorded_as_30_pct": round((days == 30).mean() * 100, 2),
    "days_27_to_30_pct": round(
        days.between(27, 30).mean() * 100, 2
    ),
    "days_25_to_30_pct": round(
        days.between(25, 30).mean() * 100, 2
    ),
    "patients_with_bmi": df["BMI_CLOSEST30D"].notna().sum(),
    "patients_with_both": df[
        ["BMI_CLOSEST30D", "DTBMI_30D"]
    ].notna().all(axis=1).sum(),
}])

summary.to_csv(
    output_dir / "bmi_timing_summary.csv",
    index=False,
)

# Count measurements for each recorded day.
counts = (
    days.value_counts()
    .sort_index()
    .rename_axis("recorded_day")
    .reset_index(name="patient_count")
)

counts.to_csv(
    output_dir / "bmi_timing_counts.csv",
    index=False,
)

print("=== SUMMARY ===")
print(summary.to_string(index=False))
print("\nSaved outputs/bmi_timing_summary.csv")
print("Saved outputs/bmi_timing_counts.csv")

# Plot the distribution of measurement timing.
plt.figure(figsize=(10, 5))
plt.bar(counts["recorded_day"], counts["patient_count"])
plt.xlabel("Recorded days from surgery to BMI measurement")
plt.ylabel("Number of patients")
plt.title("Distribution of BMI Follow-up Timing")
plt.xticks(range(0, 31, 2))
plt.tight_layout()
plt.savefig(
    output_dir / "bmi_timing_distribution.png",
    dpi=150,
)
plt.close()

print("Saved outputs/bmi_timing_distribution.png")

# Compare timing by follow-up status.
print("\n=== TIMING BY FOLLOW-UP STATUS ===")
print(
    df.groupby("FOLLOW_30DAYS_BAR", dropna=False)["DTBMI_30D"]
    .agg(["count", "median", "mean"])
    .round(2)
    .to_string()
)

# Compare BMI availability and values by follow-up status.
print("\n=== BMI AVAILABILITY BY FOLLOW-UP STATUS ===")
print(
    df.groupby("FOLLOW_30DAYS_BAR", dropna=False)["BMI_CLOSEST30D"]
    .agg(["count", "median", "mean"])
    .round(2)
    .to_string()
)

# Examine recorded reoperation values.
print("\n=== REOP30 VALUE COUNTS ===")
print(df["REOP30"].value_counts(dropna=False).to_string())

# Compare reoperation values across measurement-timing groups.
print("\n=== REOP30 BY MEASUREMENT TIMING ===")
timing_group = pd.cut(
    df["DTBMI_30D"],
    bins=[-1, 24, 30],
    labels=["Day 0-24", "Day 25-30"],
)

print(
    pd.crosstab(
        timing_group,
        df["REOP30"],
        margins=True,
        dropna=False,
    ).to_string()
)

