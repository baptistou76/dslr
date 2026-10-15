
import sys
import pandas as pd

from DSLR.data import FEATURES, prepare_features
from DSLR.model import Model


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/logreg_predict.py <test.csv>")
        return 1

    filename = sys.argv[1]

    if not filename.lower().endswith(".csv"):
        print("Error: the input file must be a CSV.")
        return 1

    try:
        data = pd.read_csv(filename)

        if "Index" not in data.columns:
            raise ValueError("Missing 'Index' column in the test dataset.")

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

        output = pd.DataFrame({
            "Index": data["Index"],
            "Hogwarts House": predictions,
        })

        output.to_csv("houses.csv", index=False)
        print(f"Predictions saved to houses.csv ({len(output)} rows).")
        return 0

    except (OSError, pd.errors.ParserError, ValueError, KeyError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
