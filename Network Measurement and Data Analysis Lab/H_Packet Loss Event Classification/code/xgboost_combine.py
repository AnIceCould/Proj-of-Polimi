# -*- coding: utf-8 -*-


import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import optuna
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

"""plot"""

def model_plot(model, X_test, y_pred, description: str):


    ''' # Plotting feature importance graphs
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
    shap.summary_plot(shap_values, X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    '''

    # Original confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the normalized confusion matrix
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["0 (normal)", "1(Packet loss)"])
    disp_norm.plot(values_format='.4f')
    plt.title("Normalized Confusion Matrix")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / f"confusion_matrix_{description}.png")

def evaluate_model(model, X_test, y_pred, description: str):

    # Get Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1_score = report["1"]["f1-score"]

    return recall, precision, f1_score

"""main"""

LOOKBACK_SECONDS_LIST = [5]
PREDICT_SECONDS_LIST = [1]
TEST_RATIO = 0.2
RANDOM_STATE = 42
BASE_DIR = Path("./example")

for LOOKBACK_SECONDS in LOOKBACK_SECONDS_LIST:
    for PREDICT_SECONDS in PREDICT_SECONDS_LIST:

        SAVE_DIR = BASE_DIR / f"n{LOOKBACK_SECONDS}x{PREDICT_SECONDS}" / "windowed"
        OUT_DIR = BASE_DIR / f"n{LOOKBACK_SECONDS}x{PREDICT_SECONDS}" / "xgb_mobile_out"
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        all_dfs = []
        header_saved = None

        for filename in os.listdir(SAVE_DIR):
            if filename.endswith('-mobile.csv') and filename.startswith('cpe_'):
                file_path = os.path.join(SAVE_DIR, filename)
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
        else:
            print("error")

        y = merged_df["label"]
        X = merged_df.drop(columns=["label"])
        X = merged_df.drop(columns=["label", "std_delay", "loss_ratio", "last_delay"])

        print(f"\n=== LOOKBACK={LOOKBACK_SECONDS}s, PREDICT={PREDICT_SECONDS}s ===")
        print(f"SAVE_DIR: {SAVE_DIR}")
        print(f"OUT_DIR : {OUT_DIR}")


        label = f"all-mobile-n{LOOKBACK_SECONDS}x{PREDICT_SECONDS}"

        # Data partitioning
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_RATIO,
            random_state=RANDOM_STATE, stratify=y
        )

        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        base_scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        trial_results = []

        def objective(trial):
            params = {
                "n_estimators"      : trial.suggest_int("n_estimators", 50, 600),
                "max_depth"         : trial.suggest_int("max_depth", 2, 8),
                "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample"         : trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "scale_pos_weight"  : trial.suggest_float("scale_pos_weight", 0.5, base_scale_pos_weight * 2),
                "min_child_weight"  : trial.suggest_float("min_child_weight", 0.1, 5),
                "gamma"             : trial.suggest_float("gamma", 0, 1),
                "eval_metric"       : "aucpr",
                "random_state"      : RANDOM_STATE
            }

            model = XGBClassifier(**params)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            recall_scorer = make_scorer(recall_score, pos_label=1)

            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring=recall_scorer, n_jobs=-1
            )

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            trial_results.append({
                'trial_number': trial.number,
                'mean_score': mean_score,
                'std_score': std_score,
                'params': params.copy()
            })

            print(f"Trial {trial.number}: Score={mean_score:.4f}±{std_score:.4f}")
            return mean_score

        # Run Bayesian optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler()
        )

        study.optimize(objective, n_trials=10, show_progress_bar=True)

        sorted_results = sorted(trial_results, key=lambda x: x['mean_score'], reverse=True)

        # Final training and evaluation
        best_params = study.best_params
        best_model = XGBClassifier(**best_params)
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
            "params"    : best_params
        }]).to_csv(out_csv, index=False)
        print(f"in {out_csv}")