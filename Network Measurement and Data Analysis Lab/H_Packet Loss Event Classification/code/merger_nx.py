import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("./example")
LOOKBACK_SECONDS_LIST = [5]
PREDICT_SECONDS_LIST = [1]

all_data = []

# Traverse all combinations
for lookback in LOOKBACK_SECONDS_LIST:
    for predict in PREDICT_SECONDS_LIST:
        out_dir = BASE_DIR / f"n{lookback}x{predict}" / "xgb_mobile_out"
        csv_files = list(out_dir.glob("*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df["lookback_seconds"] = lookback
                df["predict_seconds"] = predict
                all_data.append(df)
            except Exception as e:
                print(f"{csv_file}: {e}")

# Merge all data
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(BASE_DIR / "combined_results_xgb_mobile.csv", index=False)
    print(f"The merge is complete and the result has been saved as: {BASE_DIR / 'combined_results_xgb_fiber.csv'}")
else:
    print("No CSV file was found or the file reading failed.")

# Read the merged CSV file
combined_path = Path(BASE_DIR / "combined_results_xgb_mobile.csv")
df = pd.read_csv(combined_path)

pivot_table = df.pivot_table(
    index="lookback_seconds",
    columns="predict_seconds",
    values="recall",
    aggfunc="mean"
)

# Draw a heat map
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={'label': 'Recall'})
plt.title("XGBoost Recall Heatmap")
plt.xlabel("Predict Seconds")
plt.ylabel("Lookback Seconds")
plt.tight_layout()
plt.savefig(BASE_DIR / "recall_heatmap_xgb_mobile.png")
