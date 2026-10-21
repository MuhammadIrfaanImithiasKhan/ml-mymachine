import os
import json
import hashlib

import numpy as np
from sklearn.model_selection import train_test_split

import config
import utils

SEED = 42
OUTPUT_PATH = os.path.join(
    config.MODEL_DIR,
    "train_val_test_indices_v2.json"
)

if os.path.exists(OUTPUT_PATH):
    raise FileExistsError(
        f"{OUTPUT_PATH} already exists. Refusing to overwrite it."
    )

df = utils.load_dataframe("03_scored")
case_col = config.CASEID_COLUMN

if not df.index.is_unique:
    raise ValueError("DataFrame index must be unique.")

if df[case_col].isna().any() or df[case_col].duplicated().any():
    raise ValueError("CASEIDs must be present and unique.")

# Fingerprint the ordered CASEID list to document which dataset
# ordering was used to generate this split.
case_ids = df[case_col].astype(str).tolist()
fingerprint = hashlib.sha256(
    "\n".join(case_ids).encode("utf-8")
).hexdigest()

# First reserve 30% for validation + test.
train_idx, remaining_idx = train_test_split(
    df.index.to_numpy(),
    test_size=0.30,
    random_state=SEED,
    shuffle=True
)

# Split the remaining 30% evenly into validation and test.
val_idx, test_idx = train_test_split(
    remaining_idx,
    test_size=0.50,
    random_state=SEED,
    shuffle=True
)

train_set = set(map(int, train_idx))
val_set = set(map(int, val_idx))
test_set = set(map(int, test_idx))

if train_set & val_set or train_set & test_set or val_set & test_set:
    raise ValueError("Split overlap detected.")

if train_set | val_set | test_set != set(map(int, df.index)):
    raise ValueError("Split does not cover the dataset exactly.")

payload = {
    "train_indices": sorted(map(int, train_idx)),
    "validation_indices": sorted(map(int, val_idx)),
    "test_indices": sorted(map(int, test_idx)),
    "metadata": {
        "random_seed": SEED,
        "method": "two-stage sklearn train_test_split",
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "n_rows": len(df),
        "caseid_column": case_col,
        "ordered_caseid_sha256": fingerprint
    }
}

os.makedirs(config.MODEL_DIR, exist_ok=True)

with open(OUTPUT_PATH, "x") as f:
    json.dump(payload, f)

print("Created:", OUTPUT_PATH)
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))
print("Seed:", SEED)
print("CASEID fingerprint:", fingerprint)
print("Overlap checks: passed")
print("Coverage check: passed")
