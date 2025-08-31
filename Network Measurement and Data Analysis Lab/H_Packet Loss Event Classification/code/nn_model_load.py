# -*- coding: utf-8 -*-

import warnings
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import precision_score, recall_score
import joblib
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

"""load_file"""
BASE_DIR = Path("./example")
SAVE_DIR2 = BASE_DIR / f"n5x1" / "windowed/dir2"
SAVE_DIR1 = BASE_DIR / f"n5x1" / "windowed/dir1"
OUT_DIR = BASE_DIR / f"n5x1" / "fl_mobile_nn_out/fed"

def load_file(filepath):
    all_dfs = []
    header_saved = None

    for filename in os.listdir(filepath):
        if filename.endswith('-mobile.csv') and filename.startswith('cpe_a'):
            file_path = os.path.join(filepath, filename)
            try:
                df = pd.read_csv(file_path)
                if header_saved is None:
                    header_saved = df.columns
                df_no_header = pd.read_csv(file_path, skiprows=1, header=None)
                all_dfs.append(df_no_header)
                print(f"{filename}")
            except Exception as e:
                print(f"{filename}: {e}")

    if all_dfs and header_saved is not None:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.columns = header_saved


    y = merged_df["label"]
    X = merged_df.drop(columns=["label", "std_delay", "loss_ratio", "last_delay"])

    print(f"SAVE_DIR: {filepath}")

    return X, y

print(f"OUT_DIR : {OUT_DIR}")

X_test, y_test = load_file(SAVE_DIR2)
scaler = StandardScaler()
scaler = joblib.load(OUT_DIR / "scaler_a.pkl")
X_test = scaler.fit_transform(X_test)
label = f"nn_a-mobile-n5x1_fl_fed"

# Load the model
model = load_model(OUT_DIR / "controller_fed_model_nn.h5")
y_pred_prob = model.predict(X_test).ravel()
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
best_recall = -1
best_threshold = 0.5

for p, r, t in zip(precisions, recalls, thresholds):
    if p >= 0.2 and r > best_recall:
        best_recall = r
        best_threshold = t

y_pred = (y_pred_prob > best_threshold).astype(int)

f1 = f1_score(y_test, y_pred, zero_division=0)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)

result_row = {
    "F1 Score": f1,
    "Precision": prec,
    "Recall": rec,
    "Best Threshold": best_threshold
}

cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.4f', cmap='Blues')
plt.title(f"Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(OUT_DIR / f"confusion_matrix_{label}.png")

results_df = pd.DataFrame([result_row])
print("\n Summary of All Link Evaluations:")
print(results_df)
out_csv = OUT_DIR / f"{label}.csv"
results_df.to_csv(out_csv, index=False)