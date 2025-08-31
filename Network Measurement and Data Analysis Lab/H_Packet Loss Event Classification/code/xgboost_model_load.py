# -*- coding: utf-8 -*-


import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import pickle


warnings.filterwarnings("ignore")

"""plot"""

def model_plot(model, X_test, y_pred, description: str):

    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized,
                                      display_labels=["0 (normal)", "1(Packet loss)"])
    disp_norm.plot(values_format='.4f')
    plt.title("Normalized Confusion Matrix")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / f"confusion_matrix_{description}.png")

def evaluate_model(model, X_test, y_pred, description: str):

    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1_score = report["1"]["f1-score"]

    return recall, precision, f1_score

"""load_file"""
BASE_DIR = Path("./example")
SAVE_DIR2 = BASE_DIR / f"n5x1" / "windowed/dir2"
SAVE_DIR1 = BASE_DIR / f"n5x1" / "windowed/dir1"
OUT_DIR = BASE_DIR / f"n5x1" / "fl_mobile_xgb_out/fed"

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

X_train, y_train = load_file(SAVE_DIR1)
X_test, y_test = load_file(SAVE_DIR2)
label = f"xgb_a-mobile-n5x1_fl_all"

# load model
with open(OUT_DIR / 'controller_model_xgb.pkl', 'rb') as f:
    best_model = pickle.load(f)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

# evaluation
best_recall, best_precision, best_f1_score = evaluate_model(best_model, X_test, y_pred, description=label)
print(f"best_recall: {best_recall:.4f}")
print(f"best_precision: {best_precision:.4f}")
print(f"best_f1_score: {best_f1_score:.4f}")

# plot
model_plot(best_model, X_test, y_pred, description=label)

# output
out_csv = OUT_DIR / f"{label}.csv"
pd.DataFrame([{
    "recall"    : best_recall,
    "precision" : best_precision,
    "f1_score"  : best_f1_score,
}]).to_csv(out_csv, index=False)
print(f"{out_csv}")