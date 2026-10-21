#!/usr/bin/env python3
"""Leakage audit for obesity-digital-twin-ml. Run at repo root:
python audit_training_leakage_v2.py

Writes outputs/training_leakage_audit_v2.txt and .json.
Static code checks are evidence, not a formal proof; REVIEW items need inspection.
"""
import json, re, sys
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "outputs/04_final_scored_v2.parquet"
SPLIT = ROOT / "models/train_val_test_indices_v2.json"
SCRIPTS = {
    "MLP training": ROOT / "05_train_model_v2.py",
    "Baseline models": ROOT / "08_baseline_models_v2.py",
    "XGBoost": ROOT / "09_xgboost_models_v2.py",
    "Paired comparison": ROOT / "10_paired_model_comparison_v2.py",
}
METRICS = [
    ROOT / "models/training_metrics_v2.json",
    ROOT / "models/xgboost_metrics_v2.json",
    ROOT / "models/model_comparison_metrics_v2.json",
]
checks = []

def add(name, status, evidence, note=""):
    checks.append({"check": name, "status": status, "evidence": evidence, "note": note})

def read(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None

def main():
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    print("=" * 76)
    print("TRAINING / VALIDATION / TEST LEAKAGE AUDIT")
    print("Run:", now)
    print("Repo:", ROOT)
    print("=" * 76)

    # Validate the fixed split against the scored data.
    if not SPLIT.exists():
        add("Permanent split file", "FAIL", f"Missing {SPLIT}")
    elif not DATA.exists():
        add("Scored dataset", "FAIL", f"Missing {DATA}")
    else:
        try:
            import pandas as pd
            df = pd.read_parquet(DATA)
            s = json.loads(read(SPLIT))
            tr, va, te = ([int(i) for i in s[k]] for k in
                          ("train_indices", "validation_indices", "test_indices"))
            sets = [set(tr), set(va), set(te)]
            add("Dataset and split loaded", "INFO",
                f"rows={len(df):,}; train={len(tr):,}; validation={len(va):,}; test={len(te):,}")
            overlaps = [len(sets[0]&sets[1]), len(sets[0]&sets[2]), len(sets[1]&sets[2])]
            add("No split overlap", "PASS" if not any(overlaps) else "FAIL",
                f"train∩validation={overlaps[0]}; train∩test={overlaps[1]}; validation∩test={overlaps[2]}")
            coverage = (len(tr)+len(va)+len(te) == len(df)
                        and len(set(tr+va+te)) == len(df)
                        and set(tr+va+te) == set(range(len(df))))
            add("Every dataset row assigned exactly once", "PASS" if coverage else "FAIL",
                f"assigned={len(tr)+len(va)+len(te):,}; unique={len(set(tr+va+te)):,}; expected={len(df):,}")
            ratios = [len(tr)/len(df), len(va)/len(df), len(te)/len(df)]
            ratio_ok = all(abs(x-y) < .005 for x,y in zip(ratios, (.70,.15,.15)))
            add("Split proportions approximately 70/15/15", "PASS" if ratio_ok else "REVIEW",
                f"train={ratios[0]:.3%}; validation={ratios[1]:.3%}; test={ratios[2]:.3%}")
        except Exception as e:
            add("Load/validate split", "FAIL", f"{type(e).__name__}: {e}")

    loaded = {}
    for label, path in SCRIPTS.items():
        content = read(path)
        if content is None:
            add(f"{label} script exists", "REVIEW", f"Missing {path.name}")
        else:
            loaded[label] = content
            add(f"{label} script exists", "PASS", path.name)

    # Look for the explicit train-only patterns known to exist in this pipeline.
    for label in ("MLP training", "Baseline models", "XGBoost"):
        text = loaded.get(label)
        if not text:
            continue
        checks_to_run = [
            ("severity imputation uses training rows",
             r"train_values\s*=\s*X\.loc\[\s*train_idx.*?median\(",
             "Looks for X.loc[train_idx] followed by median calculation."),
            ("HGT/OPYEAR mean and std use training rows",
             r"train_values\s*=\s*pd\.to_numeric\(\s*df\.loc\[\s*train_idx.*?train_mean\s*=\s*train_values\.mean\(\).*?train_std\s*=\s*train_values\.std\(\)",
             "Looks for mean/std derived from df.loc[train_idx]."),
            ("StandardScaler fit is training-only",
             r"scaler\.fit\(\s*X(?:_scaled)?\.loc\[\s*train_idx\s*\]\s*\)",
             "Looks for scaler.fit(X.loc[train_idx]); transforming all rows afterwards is expected."),
        ]
        for title, pattern, note in checks_to_run:
            found = bool(re.search(pattern, text, re.I | re.S))
            add(f"{label}: {title}", "PASS" if found else "REVIEW",
                "Expected train-only code pattern found." if found else "Expected pattern not found; inspect code.", note)

        suspicious = []
        for pattern, desc in [
            (r"scaler\.fit_transform\(\s*X\s*\)", "scaler.fit_transform(X)"),
            (r"scaler\.fit\(\s*X\s*\)", "scaler.fit(X) on all rows"),
            (r"train_test_split\s*\(", "new train_test_split call"),
        ]:
            if re.search(pattern, text, re.I):
                suspicious.append(desc)
        add(f"{label}: full-data scaler / replacement split scan",
            "REVIEW" if suspicious else "PASS",
            "; ".join(suspicious) if suspicious else "No obvious common pattern found.",
            "Any flagged match may be a comment or unrelated code; inspect context.")

    # Validation-based tuning indicators.
    for label, text in loaded.items():
        has_val = bool(re.search(r"\b(val_idx|val_rows|X_val|y_val|validation)\b", text, re.I))
        has_select = bool(re.search(
            r"(optuna|study\.optimize|best_params|best_trial|selected_alpha|selected_config|"
            r"best_iteration|early_stopping|validation.*rmse|val.*rmse|val_loss)", text, re.I))
        add(f"{label}: validation/tuning code indicators",
            "PASS" if has_val and has_select else "REVIEW",
            f"validation reference={has_val}; tuning/selection reference={has_select}",
            "This is a code-pattern check, not proof that test results never influenced choices.")

    # Conservative test-use scan. Same-line hits are flagged for manual context review.
    for label, text in loaded.items():
        suspicious = []
        for n, line in enumerate(text.splitlines(), 1):
            x = line.strip()
            if x and not x.startswith("#") and re.search(r"\b(test_idx|test_rows|X_test|y_test)\b", x, re.I) and re.search(
                r"(fit|optimi|search|trial|train|select|best_)", x, re.I):
                suspicious.append(f"{n}: {x[:160]}")
        add(f"{label}: test references near fitting/tuning",
            "REVIEW" if suspicious else "PASS",
            "\n".join(suspicious[:10]) if suspicious else "No same-line suspicious pattern found.",
            "Final evaluation on test data is allowed; choosing preprocessing, thresholds, or hyperparameters from test results is not.")

    # Read saved metric JSONs as supporting artifacts.
    for path in METRICS:
        if not path.exists():
            add(f"Metrics artifact {path.name}", "REVIEW", "Not found")
        else:
            try:
                obj = json.loads(read(path))
                has_test = "test" in json.dumps(obj).lower() or "test_set" in json.dumps(obj).lower()
                add(f"Metrics artifact {path.name}", "INFO" if has_test else "REVIEW",
                    f"Valid JSON; top-level keys={list(obj.keys())}; test metrics indicator={has_test}")
            except Exception as e:
                add(f"Metrics artifact {path.name}", "REVIEW", f"JSON parse error: {e}")

    counts = {s: sum(c["status"] == s for c in checks) for s in ("PASS","REVIEW","FAIL","INFO")}
    overall = "FAIL" if counts["FAIL"] else ("REVIEW REQUIRED" if counts["REVIEW"] else "PASS")
    report = {
        "created_at": now, "repository_root": str(ROOT), "dataset": str(DATA),
        "split_file": str(SPLIT), "overall": overall, "summary": counts, "checks": checks,
        "limitation": "Static pattern checks provide evidence, not formal proof. Manually review REVIEW items and inspect code paths."
    }
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    txt_path = out / "training_leakage_audit_v2.txt"
    json_path = out / "training_leakage_audit_v2.json"
    lines = [ "TRAINING / VALIDATION / TEST LEAKAGE AUDIT", f"Run: {now}",
              f"Repo: {ROOT}", f"OVERALL: {overall}", f"Summary: {counts}", "",
              report["limitation"], "" ]
    for c in checks:
        lines.append(f"[{c['status']}] {c['check']}")
        lines.append("  Evidence: " + c["evidence"].replace("\n", " | "))
        if c["note"]:
            lines.append("  Note: " + c["note"])
        lines.append("")
        print(f"[{c['status']:<6}] {c['check']}")
        print("         " + c["evidence"].replace("\n", " | "))
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 76)
    print(f"OVERALL: {overall} | PASS={counts['PASS']} REVIEW={counts['REVIEW']} FAIL={counts['FAIL']} INFO={counts['INFO']}")
    print("Text report:", txt_path)
    print("JSON report:", json_path)
    print("=" * 76)
    if counts["FAIL"]:
        sys.exit(1)

if __name__ == "__main__":
    main()
