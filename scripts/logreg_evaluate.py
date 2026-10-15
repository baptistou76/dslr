
import sys
import numpy as np
import pandas as pd

from DSLR.data import FEATURES, prepare_features
from DSLR.model import Model


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/logreg_evaluate.py <training.csv>")
        return 1

    filename = sys.argv[1]

    if not filename.lower().endswith(".csv"):
        print("Error: the input file must be a CSV.")
        return 1

    try:
        data = pd.read_csv(filename)

        if "Hogwarts House" not in data.columns:
            raise ValueError("Missing 'Hogwarts House' column.")

        data = data.dropna(subset=["Hogwarts House"]).copy()

        if data.empty:
            raise ValueError("No labeled samples were found.")

        y_true = data["Hogwarts House"].astype(str).to_numpy()

        if not np.isin(y_true, Model.HOUSES).all():
            raise ValueError("The dataset contains an unknown house.")

        model = Model().load("logreg_model.npz")
        features = model.features or FEATURES

        if model.imputation_means is None:
            raise ValueError(
                "The model has no imputation means. Retrain it first."
            )

        X, _ = prepare_features(
            data,
            features,
            model.imputation_means,
        )

        predictions = model.predict(X)
        accuracy = np.mean(predictions == y_true) * 100

        print(f"Overall accuracy: {accuracy:.2f}%")
        print("\nAccuracy by house:")

        for house in Model.HOUSES:
            mask = y_true == house
            count = int(mask.sum())

            if count == 0:
                print(f"{house}: no samples")
                continue

            house_accuracy = np.mean(
                predictions[mask] == y_true[mask]
            ) * 100

            print(
                f"{house}: {house_accuracy:.2f}% "
                f"({count} samples)"
            )

        return 0

    except (OSError, pd.errors.ParserError, ValueError, KeyError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
