# -*- coding: utf-8 -*-

import warnings
from pathlib import Path
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

DATA_DIR1 = Path("./example/1st_capture")
DATA_DIR2 = Path("./example/2nd_capture")
BASE_DIR = Path("./example")

LOOKBACK_SECONDS = 5
PREDICT_SECONDS = 1

def preprocess_single_data(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    stem = filepath.stem
    try:
        src_dst, method = stem.rsplit("-", 1)
        src, dst = src_dst.split("-", 1)
    except ValueError:
        raise ValueError(f"error: {filepath.name}")

    df["src"] = src
    df["dst"] = dst
    df["method"] = method

    df["packet_loss"] = (df["delay_ms"] == -1).astype(int)
    df.loc[df['packet_loss'] == 1, 'delay_ms'] = 999

    return df


def sliding_window_samples(df: pd.DataFrame, lookback: int, predict: int, original_filepath: Path, save_dir: Path):

    delay_array = df["delay_ms"].values
    loss_array = df["packet_loss"].values
    total_rows = len(delay_array)

    n_samples = total_rows - lookback - predict + 1
    if n_samples <= 0:
        return pd.DataFrame(), pd.Series(dtype=int)

    def sliding_window_view(arr, window_size, step=1):
        shape = ((arr.size - window_size) // step + 1, window_size)
        strides = (arr.strides[0] * step, arr.strides[0])
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    history_delay_windows = sliding_window_view(delay_array, lookback)[:n_samples]
    history_loss_windows = sliding_window_view(loss_array, lookback)[:n_samples]

    future_start_idx = lookback
    future_loss_windows = sliding_window_view(loss_array[future_start_idx:], predict)[:n_samples]

    y = np.any(future_loss_windows == 1, axis=1).astype(int)

    n_features = 11 + lookback
    X = np.empty((n_samples, n_features))

    X[:, 0] = np.mean(history_delay_windows, axis=1)    # mean_delay
    X[:, 1] = np.std(history_delay_windows, axis=1)     # std_delay
    X[:, 2] = np.min(history_delay_windows, axis=1)     # min_delay
    X[:, 3] = np.median(history_delay_windows, axis=1)  # mid_delay
    X[:, 4] = np.max(history_delay_windows, axis=1)     # max_delay
    X[:, 5] = history_delay_windows[:, -1]              # last_delay

    x_trend = np.arange(lookback)
    slopes = []
    for i in range(n_samples):
        valid_mask = history_delay_windows[i] != -1
        if np.sum(valid_mask) > 1:
            valid_delays = history_delay_windows[i][valid_mask]
            valid_x = x_trend[valid_mask]
            slope = np.polyfit(valid_x, valid_delays, 1)[0]
        else:
            slope = 0
        slopes.append(slope)
    X[:, 6] = np.array(slopes)  # slope_delay

    X[:, 7] = np.mean(history_loss_windows == 1, axis=1)  # loss_ratio

    X[:, 8] = np.mean(history_delay_windows[:, -3:], axis=1)  # mean_of_last_three

    X[:, 9] = history_delay_windows[:, -2] - history_delay_windows[:, -1]  # diff_between_last_two

    X[:, 10] = np.max(history_delay_windows, axis=1) - np.min(history_delay_windows, axis=1)

    X[:, 11:] = history_delay_windows

    feature_names = [
        "mean_delay", "std_delay", "min_delay", "mid_delay", "max_delay",
        "last_delay", "slope_delay", "loss_ratio", "mean_of_last_three",
        "diff_between_last_two", "range"
    ] + [f"delay_{j+1}" for j in range(lookback)]

    X_df = pd.DataFrame(X, columns=feature_names)
    y_sr = pd.Series(y, name="label")

    samples_df = pd.concat([X_df, y_sr], axis=1)
    samples_df = samples_df.replace([np.inf, -np.inf], np.nan)

    original_count = len(samples_df)
    samples_df = samples_df.dropna()
    cleaned_count = len(samples_df)

    X_df = samples_df.drop(columns=["label"])
    y_sr = samples_df["label"]

    save_dir.mkdir(parents=True, exist_ok=True)
    filename = original_filepath.name
    save_path = save_dir / filename

    samples_df.to_csv(save_path, index=False)

    return X_df, y_sr

def process_files_separately():

    selected_files = list(DATA_DIR1.glob("*.csv"))

    SAVE_DIR1 = BASE_DIR / f"n{LOOKBACK_SECONDS}x{PREDICT_SECONDS}" / "windowed" / "dir1"
    SAVE_DIR2 = BASE_DIR / f"n{LOOKBACK_SECONDS}x{PREDICT_SECONDS}" / "windowed" / "dir2"

    SAVE_DIR1.mkdir(parents=True, exist_ok=True)
    SAVE_DIR2.mkdir(parents=True, exist_ok=True)

    print(f"\n=== LOOKBACK={LOOKBACK_SECONDS}s, PREDICT={PREDICT_SECONDS}s ===")
    print(f"DIR1 SAVE_DIR: {SAVE_DIR1}")
    print(f"DIR2 SAVE_DIR: {SAVE_DIR2}")

    for file1 in selected_files:
        file2 = DATA_DIR2 / file1.name

        try:
            df1 = preprocess_single_data(file1)
            label1 = f"{df1['src'].iloc[0]}->{df1['dst'].iloc[0]} ({df1['method'].iloc[0]}) - DIR1"

            X1, y1 = sliding_window_samples(df1, LOOKBACK_SECONDS, PREDICT_SECONDS, file1, SAVE_DIR1)

        except Exception as e:
            print(f"{file1.name}: {e}")
            continue

        try:
            df2 = preprocess_single_data(file2)
            label2 = f"{df2['src'].iloc[0]}->{df2['dst'].iloc[0]} ({df2['method'].iloc[0]}) - DIR2"

            X2, y2 = sliding_window_samples(df2, LOOKBACK_SECONDS, PREDICT_SECONDS, file2, SAVE_DIR2)

        except Exception as e:
            print(f" {file2.name}{e}")
            continue

# main
process_files_separately()