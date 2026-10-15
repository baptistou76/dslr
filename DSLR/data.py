
import numpy as np
import pandas as pd


FEATURES = [
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying",
]


def prepare_features(data, features, imputation_means=None):
    """Convert features to numbers and replace missing values."""

    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    values = data[features].apply(pd.to_numeric, errors="coerce")

    if imputation_means is None:
        means = values.mean()
        if means.isna().any():
            invalid = means[means.isna()].index.tolist()
            raise ValueError(
                f"Cannot calculate means for empty features: {invalid}"
            )
    else:
        means = pd.Series(imputation_means, index=features, dtype=float)

    values = values.fillna(means)

    X = values.to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        raise ValueError("Features contain invalid numeric values.")

    return X, means.to_numpy(dtype=float)
