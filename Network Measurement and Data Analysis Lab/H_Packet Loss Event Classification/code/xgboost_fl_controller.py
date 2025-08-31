from sklearn.ensemble import VotingClassifier
import glob
import joblib
from pathlib import Path

# Load all CPE models
BASE_DIR = Path("./example")
OUT_DIR = BASE_DIR / "n5x1/fl_mobile_xgb_out/fed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

model_files = glob.glob("./example/n5x1/fl_mobile_xgb_out/*.pkl")
models = [(f"model_{i}", joblib.load(path)) for i, path in enumerate(model_files)]

# Build a soft voting aggregation model
controller_model = VotingClassifier(estimators=models, voting='soft')
controller_model_path = OUT_DIR / "controller_model_xgb.pkl"
joblib.dump(controller_model, controller_model_path)
print(f"{controller_model_path}")

