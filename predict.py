"""Score the latest MySuper quarter with the saved Reduced 29 Random Forest."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from transform import ENTITY_KEYS, REDUCED_FEATURE_COLS, check_columns

# Define predict.py beside etl.py and transform.py in the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "outputs" / "final_rf_reduced29_t035" / "model.joblib"
)

def main():
    # Select the prepared edition and the already fitted model.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", required=True, help="Prepared edition, e.g. 2026-06")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    data_path = (
        PROJECT_ROOT / "data" / "processed"
        / f"hist_model_mysuper_features_{args.edition}.parquet"
    )
    for label, path in [("Prepared data", data_path), ("Saved model", args.model)]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    # Read the model bundle based on baseline_modelling.ipynb.
    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or not {"model", "specification"}.issubset(bundle):
        raise ValueError("Expected a saved bundle containing model and specification.")
    model = bundle["model"]
    specification = bundle["specification"]
    feature_cols = list(specification["features"])
    threshold = float(specification["threshold"])

    # Preserve the selected specification and original training column order.
    if feature_cols != REDUCED_FEATURE_COLS or not np.isclose(threshold, 0.35):
        raise ValueError("Expected the selected Reduced 29 model at threshold 0.35.")
    if list(model.feature_names_in_) != feature_cols:
        raise ValueError("Saved model feature order differs from its specification.")
    if specification["sklearn_version"] != sklearn.__version__:
        raise ValueError(
            f"Model saved with scikit-learn {specification['sklearn_version']}; "
            f"current version is {sklearn.__version__}. Use the training environment."
        )
    if set(model.classes_) != {0, 1}:
        raise ValueError("Expected a fitted binary classifier with classes 0 and 1.")
