
import sys
import pandas as pd

from DSLR.data import FEATURES, prepare_features
from DSLR.model import Model


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/logreg_train.py <training.csv>")
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
            raise ValueError("No rows with a known house were found.")

        houses = data["Hogwarts House"].astype(str)

        if not houses.isin(Model.HOUSES).all():
            raise ValueError("The dataset contains an unknown house.")

        X, means = prepare_features(data, FEATURES)
        y = houses.to_numpy()

        model = Model()
        model.fit(X, y)

        model.features = FEATURES.copy()
        model.imputation_means = means
        model.save("logreg_model.npz")

        print(f"Training completed with {len(data)} samples.")
        return 0

    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
