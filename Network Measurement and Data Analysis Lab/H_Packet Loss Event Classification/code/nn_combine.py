import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path("./example")
LOOKBACK_SECONDS_LIST = [5]
PREDICT_SECONDS_LIST = [1]

for LOOKBACK_SECONDS in LOOKBACK_SECONDS_LIST:
    for PREDICT_SECONDS in PREDICT_SECONDS_LIST:

        SAVE_DIR = BASE_DIR / f"n{LOOKBACK_SECONDS}x{PREDICT_SECONDS}" / "windowed"
        OUT_DIR = BASE_DIR / f"n{LOOKBACK_SECONDS}x{PREDICT_SECONDS}" / "nn_mobile_out"
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        # Store all data (remove the header row)
        all_dfs = []
        header_saved = None

        for filename in os.listdir(SAVE_DIR):
            if filename.endswith('-mobile.csv') and filename.startswith('cpe_'):
                file_path = os.path.join(SAVE_DIR, filename)
                try:
                    # Read the entire file, preserving the column names to extract the header once
                    df = pd.read_csv(file_path)
                    if header_saved is None:
                        header_saved = df.columns  # Save column names once
                    # Read the file and skip the first line (the column name line)
                    df_no_header = pd.read_csv(file_path, skiprows=1, header=None)
                    all_dfs.append(df_no_header)
                    print(f"Loaded and stripped headers: {filename}")
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")

        # Merge all data
        if all_dfs and header_saved is not None:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            # Add back the column names
            merged_df.columns = header_saved
        else:
            print("No valid data was found or the file is empty.")

        all_results = {}

        X = df.drop(columns=["label"])
        X = X.drop(columns=["std_delay", "loss_ratio", "last_delay"])
        y = df["label"].values

        print(f"\n=== Processing configuration: LOOKBACK={LOOKBACK_SECONDS}s, PREDICT={PREDICT_SECONDS}s ===")
        print(f"SAVE_DIR: {SAVE_DIR}")
        print(f"OUT_DIR : {OUT_DIR}")

        label = f"all-mobile-n{LOOKBACK_SECONDS}x{PREDICT_SECONDS}"

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Class weights
        weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        weights = dict(enumerate(weights))
        weights[1] *= 1.5  # Strengthen positive class penalty and improve recall

        # Model structure
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0008), loss='binary_crossentropy', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=50, batch_size=128,
                            validation_data=(X_val, y_val),
                            class_weight=weights,
                            callbacks=[early_stop], verbose=0)

        # Automatic threshold selection
        y_pred_prob = model.predict(X_val).ravel()
        # Calculate PR curve
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_prob)

        # Traverse all thresholds and find the point with the largest recall when precision >= 0.2
        best_recall = -1
        best_threshold = 0.5

        for p, r, t in zip(precisions, recalls, thresholds):
            if p >= 0.2 and r > best_recall:
                best_recall = r
                best_threshold = t

        y_pred = (y_pred_prob > best_threshold).astype(int)

        f1 = f1_score(y_val, y_pred, zero_division=0)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)

        result_row = {
            "F1 Score": f1,
            "Precision": prec,
            "Recall": rec,
            "Best Threshold": best_threshold
        }

        cm = confusion_matrix(y_val, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f"Normalized Confusion Matrix - {label}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_matrix_{label}.png")

        # Output results
        results_df = pd.DataFrame([result_row])
        print("\n Summary of All Link Evaluations:")
        print(results_df)
        out_csv = OUT_DIR / f"{label}.csv"
        results_df.to_csv(out_csv, index=False)
