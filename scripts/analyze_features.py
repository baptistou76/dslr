
import pandas as pd
import numpy as np
import sys

from scripts.logreg_train import FEATURES, HOUSES


def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide a CSV file.")

        filename = sys.argv[1]

        if not filename.lower().endswith(".csv"):
            raise ValueError("The file must be a CSV file.")

        data = pd.read_csv(filename)
        print("CSV file successfully loaded.")

        if "Hogwarts House" not in data.columns:
            raise ValueError("Missing column: Hogwarts House.")

        for feature in FEATURES:
            if feature not in data.columns:
                raise ValueError(f"Missing feature: {feature}.")

        data = data.dropna(subset=["Hogwarts House"])

        # Calculate statistics for each feature and each house.
        results = []

        for feature in FEATURES:
            house_means = []
            house_stds = []

            for house in HOUSES:
                values = data.loc[
                    data["Hogwarts House"] == house, feature
                ].to_numpy(dtype=float)

                mean = calculate_mean(values)
                std = calculate_std(values)

                house_means.append(mean)
                house_stds.append(std)

            # Difference between the highest and lowest house means.
            mean_gap = max(house_means) - min(house_means)

            # Average standard deviation across the houses.
            average_std = sum(house_stds) / len(house_stds)

            # Relative separation: larger values suggest more separation.
            if average_std > 0:
                separation = mean_gap / average_std
            else:
                separation = 0.0

            results.append({
                "Feature": feature,
                "Mean gap": mean_gap,
                "Average std": average_std,
                "Separation": separation
            })

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="Separation",
            ascending=False
        )

        print("\n--- Feature analysis ---")
        print(
            results_df.to_string(
                index=False,
                formatters={
                    "Mean gap": "{:.3f}".format,
                    "Average std": "{:.3f}".format,
                    "Separation": "{:.3f}".format
                }
            )
        )

        print("\n--- House means for the 5 most separated features ---")

        top_features = results_df.head(5)["Feature"].tolist()

        for feature in top_features:
            print(f"\n{feature}")

            for house in HOUSES:
                values = data.loc[
                    data["Hogwarts House"] == house, feature
                ].to_numpy(dtype=float)

                mean = calculate_mean(values)
                print(f"  {house}: {mean:.3f}")

    except FileNotFoundError:
        print("Error: file not found.")

    except pd.errors.EmptyDataError:
        print("Error: the CSV file is empty.")

    except (ValueError, ZeroDivisionError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
