# -*- coding: utf-8 -*-

import warnings
from pathlib import Path
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

# Path
DATA_DIR1 = Path("./example/1st_capture")
DATA_DIR2 = Path("./example/2nd_capture")
BASE_DIR = Path("./example")

LOOKBACK_SECONDS_LIST = [5]
PREDICT_SECONDS_LIST = [1]

"""Preprocessing"""

def preprocess_data(filepath1: Path, filepath2: Path) -> pd.DataFrame:

    # Read two filespath
    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)

    stem1 = filepath1.stem
    stem2 = filepath2.stem
    try:
        src_dst1, method1 = stem1.rsplit("-", 1)
        src1, dst1 = src_dst1.split("-", 1)
        src_dst2, method2 = stem2.rsplit("-", 1)
        src2, dst2 = src_dst2.split("-", 1)

        if src1 != src2 or dst1 != dst2 or method1 != method2:
            raise ValueError(f"File name mismatch:{filepath1.name} and {filepath2.name}")
    except ValueError:
        raise ValueError(f"The file name format is incorrect and the path information cannot be parsed:{filepath1.name} or {filepath2.name}")

    # Merge two DataFrames
    df = pd.concat([df1, df2], ignore_index=True)

    df["src"] = src1
    df["dst"] = dst1
    df["method"] = method1

    # Added packet_loss tag: 1 means packet loss (delay_ms == -1)
    df["packet_loss"] = (df["delay_ms"] == -1).astype(int)
    # Set the delay_ms of the packet loss point to 999
    df.loc[df['packet_loss'] == 1, 'delay_ms'] = 999

    print(f"Total number of rows of original data after merging:{len(df)}")
    print(f"The number of original packet loss points after merging: {df['packet_loss'].sum()}")

    return df

"""Sliding Window Sampling"""

def sliding_window_samples(df: pd.DataFrame, lookback: int, predict: int, original_filepath: Path):

    delay_array = df["delay_ms"].values
    loss_array = df["packet_loss"].values
    total_rows = len(delay_array)

    n_samples = total_rows - lookback - predict + 1
    if n_samples <= 0:
        print("The data length is not enough to generate a sample")
        return pd.DataFrame(), pd.Series(dtype=int)

    print(f"Starting vectorization of {n_samples} samples...")

    # Create a sliding window view
    def sliding_window_view(arr, window_size, step=1):
        """Create a sliding window view of an array"""
        shape = ((arr.size - window_size) // step + 1, window_size)
        strides = (arr.strides[0] * step, arr.strides[0])
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    # Historical window data (n_samples, lookback)
    history_delay_windows = sliding_window_view(delay_array, lookback)[:n_samples]
    history_loss_windows = sliding_window_view(loss_array, lookback)[:n_samples]

    # Future window data (n_samples, predict)
    future_start_idx = lookback
    future_loss_windows = sliding_window_view(loss_array[future_start_idx:], predict)[:n_samples]

    # Vectorized calculation tags
    y = np.any(future_loss_windows == 1, axis=1).astype(int)

    # Vectorized feature calculation
    n_features = 11 + lookback
    X = np.empty((n_samples, n_features))

    # Basic statistics
    X[:, 0] = np.mean(history_delay_windows, axis=1)    # mean_delay
    X[:, 1] = np.std(history_delay_windows, axis=1)     # std_delay
    X[:, 2] = np.min(history_delay_windows, axis=1)     # min_delay
    X[:, 3] = np.median(history_delay_windows, axis=1)  # mid_delay
    X[:, 4] = np.max(history_delay_windows, axis=1)     # max_delay
    X[:, 5] = history_delay_windows[:, -1]              # last_delay

    # Trend slope
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


    X[:, 10] = np.max(history_delay_windows, axis=1) - np.min(history_delay_windows, axis=1) # range

    # Original delay value
    X[:, 11:] = history_delay_windows

    # Create DataFrame
    feature_names = [
        "mean_delay", "std_delay", "min_delay", "mid_delay", "max_delay",
        "last_delay", "slope_delay", "loss_ratio", "mean_of_last_three",
        "diff_between_last_two", "range"
    ] + [f"delay_{j+1}" for j in range(lookback)]

    X_df = pd.DataFrame(X, columns=feature_names)
    y_sr = pd.Series(y, name="label")

    # Data cleaning
    samples_df = pd.concat([X_df, y_sr], axis=1)
    samples_df = samples_df.replace([np.inf, -np.inf], np.nan)

    original_count = len(samples_df)
    samples_df = samples_df.dropna()
    cleaned_count = len(samples_df)

    X_df = samples_df.drop(columns=["label"])
    y_sr = samples_df["label"]

    print(f"\nVectorization completed: {cleaned_count} items (after cleaning)")
    print(f"Delete incomplete rows:{original_count - cleaned_count}")

    # Save the results
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    filename = original_filepath.name
    save_path = SAVE_DIR / filename

    samples_df.to_csv(save_path, index=False)
    print(f"The cleaned sliding window samples have been saved to {save_path}")

    return X_df, y_sr

"""Main function for data processing"""

selected_files = list(DATA_DIR1.glob("*.csv"))

for LOOKBACK_SECONDS in LOOKBACK_SECONDS_LIST:
    for PREDICT_SECONDS in PREDICT_SECONDS_LIST:

        # Build the save path
        SAVE_DIR = BASE_DIR / f"n{LOOKBACK_SECONDS}x{PREDICT_SECONDS}" / "windowed"
        OUT_DIR = BASE_DIR / f"n{LOOKBACK_SECONDS}x{PREDICT_SECONDS}" / "out"

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing configuration:LOOKBACK={LOOKBACK_SECONDS}s, PREDICT={PREDICT_SECONDS}s ===")
        print(f"SAVE_DIR: {SAVE_DIR}")
        print(f"OUT_DIR : {OUT_DIR}")

        for file1 in selected_files:
            file2 = DATA_DIR2 / file1.name
            print(f"\nProcesses file pairs: {file1.name} (from {file1.parent.name}) and {file2.name} (from {file2.parent.name})")

            df = preprocess_data(file1, file2)
            label = f"{df['src'].iloc[0]}->{df['dst'].iloc[0]} ({df['method'].iloc[0]})"

            X, y = sliding_window_samples(df, LOOKBACK_SECONDS, PREDICT_SECONDS, file1)

            print(f"Feature data for {label} has been saved")
            print(f"- Total number of samples:{len(X)}")
            print(f"- Number of positive samples:{np.sum(y)}")
            print(f"- Number of negative samples:{len(y) - np.sum(y)}")