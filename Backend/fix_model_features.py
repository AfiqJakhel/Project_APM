import joblib
from pathlib import Path

MODEL_DIR = Path("machine_learning/output/expanding_window")

for horizon in ["h1", "h3", "h7"]:
    model_files = sorted(MODEL_DIR.glob(f"model_final_{horizon}_*.pkl"), reverse=True)

    if not model_files:
        print(f"Model {horizon} tidak ditemukan")
        continue

    model_path = model_files[0]
    print(f"\nProcessing: {model_path.name}")

    # Load model
    model = joblib.load(model_path)

    # Check if has feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        print(f"   Feature names found: {len(model.feature_names_in_)} features")
        print(f"   Removing feature_names_in_...")
        delattr(model, 'feature_names_in_')

    # Overwrite original
    joblib.dump(model, model_path)
    print(f"   Saved to: {model_path.name}")

print("\nDone!")
