# Training / Validation / Test Leakage Manual Review (v2)

Review date: 2026-10-22

## Scope

Reviewed the four REVIEW flags in
`outputs/training_leakage_audit_v2.txt` and inspected relevant code in:

- `05_train_model_v2.py`
- `08_baseline_models_v2.py`
- `09_xgboost_models_v2.py`
- `10_paired_model_comparison_v2.py`

## Findings

1. **MLP:** Train-only normalization, severity imputation, and scaler fitting
   were observed. The training loop selects and restores the checkpoint with
   the lowest validation loss. Test evaluation occurs separately afterward.

2. **Baseline models:** Ridge alpha and Random Forest configurations are
   selected using validation RMSE. The selected models are then evaluated
   on the test set.

3. **XGBoost:** Training uses the training split, early stopping monitors
   validation RMSE, and final test metrics are calculated after fitting.

4. **Paired comparison:** Random Forest candidates are selected using
   validation RMSE before predictions are generated for the test set.

## Split integrity

The original audit reports 360,720 rows split into 252,504 training,
54,108 validation, and 54,108 test rows, with no overlap and complete
row assignment.

## Conclusion

No confirmed leakage was identified in the inspected code paths. The four
static audit REVIEW flags appear explainable by legitimate split-integrity
checks or final test-set evaluation.

This is a manual code-path review, not a formal proof that every possible
leakage pathway is absent. The original audit report is preserved unchanged.

## Remaining caveats

- Confirm the complete execution path before making broader claims about
  leakage absence.
- Test metrics must not be used to choose features, thresholds, models,
  or hyperparameters.
- This note documents code review findings; it does not independently
  reproduce training or model metrics.
