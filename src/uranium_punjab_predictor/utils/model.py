from pathlib import Path

import joblib
import pandas as pd


def load_model(path: str):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Model file not found at {path}")
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def prepare_features(district, lat, lon, well_depth):
    """
    Build input DataFrame compatible with model. Adjust columns if your model changes.
    """
    # Minimal, example features. Update as needed for real model.
    feature_dict = {
        "district": [district],
        "latitude": [lat],
        "longitude": [lon],
        "well_depth": [well_depth],
    }
    return pd.DataFrame(feature_dict)
