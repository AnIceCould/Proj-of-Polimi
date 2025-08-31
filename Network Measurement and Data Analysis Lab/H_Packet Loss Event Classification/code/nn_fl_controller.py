from tensorflow.keras.models import load_model
from glob import glob
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path


BASE_DIR = Path("./example")
SAVE_DIR1 = BASE_DIR / f"n5x1" / "windowed/dir1"
OUT_DIR = BASE_DIR / "n5x1/fl_mobile_nn_out"

# Load all model paths
model_paths = glob(str(OUT_DIR / "*.h5"))
print(f"Found the number of models: {len(model_paths)}")

# Load the weight list
all_weights = [load_model(path).get_weights() for path in model_paths]

# Aggregation weight
avg_weights = []
for weights_tuple in zip(*all_weights):
    avg_layer = np.mean(weights_tuple, axis=0)
    avg_weights.append(avg_layer)

# Build a new model with the same structure as the original model
def create_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0008), loss='binary_crossentropy', metrics=['accuracy'])
    return model

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

X_train, y_train = load_file(SAVE_DIR1)

global_model = create_model(X_train.shape[1])
global_model.set_weights(avg_weights)
print("FedAvg")

controller_model_path = OUT_DIR / "fed/controller_fed_model_nn.h5"
global_model.save(controller_model_path)
print(f"{controller_model_path}")
