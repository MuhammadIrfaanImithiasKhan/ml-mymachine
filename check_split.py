import json
import numpy as np
import pandas as pd

with open("models/train_val_test_indices.json") as f:
    splits = json.load(f)

df = pd.read_parquet("outputs/03_scored.parquet")

names = ["train_indices", "validation_indices", "test_indices"]
arrays = [np.asarray(splits[name], dtype=np.int64) for name in names]
combined = np.concatenate(arrays)

print("Dataset rows:", len(df))
print("Split sizes:", [len(a) for a in arrays])
print("Unique indices:", len(np.unique(combined)))

print("Train/validation overlap:", len(set(arrays[0]) & set(arrays[1])))
print("Train/test overlap:", len(set(arrays[0]) & set(arrays[2])))
print("Validation/test overlap:", len(set(arrays[1]) & set(arrays[2])))

print("Out-of-range indices:", int(((combined < 0) | (combined >= len(df))).sum()))
print("Unassigned rows:", len(df) - len(np.unique(combined)))

print("DataFrame index type:", type(df.index).__name__)
print("DataFrame index range:", df.index.min(), "to", df.index.max())
print("Has CASEID:", "CASEID" in df.columns)
