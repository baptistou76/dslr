
import sys
import numpy as np
import pandas as pd

from DSLR.data import FEATURES, prepare_features
from DSLR.model import Model


FEATURE_GROUPS = {
    "A - All 13 features": FEATURES,
    "B - Top 8 features": [
        "Charms",
        "Flying",
        "Divination",
        "Astronomy",
        "Defense Against the Dark Arts",
        "Transfiguration",
        "History of Magic",
        "Ancient Runes",
    ],
    "C - Top 5 features": [
        "Charms",
        "Flying",
        "Divination",
        "Astronomy",
        "Defense Against the Dark Arts",
    ],
    "D - Top 8 without Flying": [
        "Charms",
        "Divination",
        "Astronomy",
        "Defense Against the Dark Arts",
        "Transfiguration",
        "History of Magic",
        "Ancient Runes",
        "Muggle Studies",
    ],
}

FOLDS = 5
SEED = 42


def create_folds(data):
    """Create five folds while preserving house representation."""

    rng = np.random.default_rng(SEED)
    house_parts = {}

    for house in Model.HOUSES:
        indices = np.flatnonzero(
            data["Hogwarts House"].to_numpy() == house
        )

        if len(indices) < FOLDS:
            raise ValueError(
                f"Not enough samples for {house}: "
                f"at least {FOLDS} are required."
            )

        rng.shuffle(indices)
        house_parts[house] = np.array_split(indices, FOLDS)

    all_indices = np.arange(len(data))
    folds = []

    for fold_number in range(FOLDS):
        validation_indices = np.concatenate([
            house_parts[house][fold_number]
            for house in Model.HOUSES
        ])

        training_indices = all_indices[
            ~np.isin(all_indices, validation_indices)
        ]

        folds.append((training_indices, validation_indices))

    return folds



def evaluate_group(data, features, folds):
    fold_accuracies = []
    fold_predictions = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        folds, start=1
    ):
        train_data = data.iloc[train_indices]
        validation_data = data.iloc[validation_indices]

        X_train, means = prepare_features(train_data, features)
        X_validation, _ = prepare_features(
            validation_data, features, means
        )

        y_train = train_data["Hogwarts House"].to_numpy()
        y_validation = validation_data["Hogwarts House"].to_numpy()

        model = Model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)

        correct = np.sum(predictions == y_validation)
        incorrect = len(y_validation) - correct
        accuracy = correct / len(y_validation) * 100

        fold_accuracies.append(accuracy)

        # Keep the predictions so we can compare configurations later.
        fold_predictions.append({
            "indices": data.index[validation_indices].to_numpy(),
            "y_true": y_validation.copy(),
            "predictions": predictions.copy(),
        })

        print(
            f"  Fold {fold_number}: {accuracy:.2f}% "
            f"({correct} correct, {incorrect} incorrect)"
        )

    overall_mean = np.mean(fold_accuracies)
    print(f"  Mean fold accuracy: {overall_mean:.2f}%")

    return overall_mean, fold_accuracies, fold_predictions



def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/logreg_validate.py <training.csv>")
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
        data["Hogwarts House"] = data["Hogwarts House"].astype(str)

        if data.empty:
            raise ValueError("No labeled samples were found.")

        if not data["Hogwarts House"].isin(Model.HOUSES).all():
            raise ValueError("The dataset contains an unknown house.")

        folds = create_folds(data)
        results = []


        for group_name, features in FEATURE_GROUPS.items():
            print(f"\n{group_name}")

            mean_accuracy, fold_accuracies, fold_predictions = (
                evaluate_group(data, features, folds)
            )

            results.append({
                "Feature group": group_name,
                "Mean accuracy (%)": mean_accuracy,
                "Fold predictions": fold_predictions,
            })

        print("\n--- Validation summary ---")

        summary = pd.DataFrame([
            {
                "Feature group": result["Feature group"],
                "Mean accuracy (%)": result["Mean accuracy (%)"],
            }
            for result in results
        ])
        print(summary.to_string(index=False))

        best = summary.loc[summary["Mean accuracy (%)"].idxmax()]
        print(
            f"\nBest group: {best['Feature group']} "
            f"({best['Mean accuracy (%)']:.2f}%)"
        )

        # Compare every configuration against configuration A.
        baseline = results[0]

        print("\n--- Prediction comparison against A ---")

        for result in results[1:]:
            print(f"\n{result['Feature group']}")

            total_different = 0
            total_shared_errors = 0
            total_baseline_only = 0
            total_group_only = 0

            for fold_number, (base, current) in enumerate(
                zip(
                    baseline["Fold predictions"],
                    result["Fold predictions"],
                ),
                start=1,
            ):
                base_predictions = base["predictions"]
                current_predictions = current["predictions"]
                y_true = base["y_true"]
                indices = base["indices"]

                different = base_predictions != current_predictions
                base_errors = base_predictions != y_true
                current_errors = current_predictions != y_true

                total_different += int(np.sum(different))
                total_shared_errors += int(
                    np.sum(base_errors & current_errors)
                )
                total_baseline_only += int(
                    np.sum(base_errors & ~current_errors)
                )
                total_group_only += int(
                    np.sum(~base_errors & current_errors)
                )

                different_indices = indices[different].tolist()

                print(
                    f"  Fold {fold_number}: "
                    f"{np.sum(different)} different predictions"
                )

                if different_indices:
                    print(
                        f"    Dataset row indices: {different_indices}"
                    )

            print(
                f"  Total different predictions: {total_different}"
            )
            print(
                f"  Same incorrectly classified samples: "
                f"{total_shared_errors}"
            )
            print(
                f"  Errors only in A: {total_baseline_only}"
            )
            print(
                f"  Errors only in this configuration: "
                f"{total_group_only}"
            )


            results.append({
                "Feature group": group_name,
                "Mean accuracy (%)": mean_accuracy,
            })

        

        return 0

    except (OSError, pd.errors.ParserError, ValueError, KeyError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
