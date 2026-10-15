
import sys

import pandas as pd

from DSLR.math import Math


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.describe <file.csv>")
        return 1

    filename = sys.argv[1]

    if not filename.lower().endswith(".csv"):
        print("Error: the file must be a CSV file.")
        return 1

    try:
        data = pd.read_csv(filename)
        print("CSV file successfully loaded.")

        if "Index" in data.columns:
            data = data.drop(columns=["Index"])

        numeric_columns = data.select_dtypes(
            include="number"
        ).columns

        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns were found.")

        results = {}

        for feature in numeric_columns:
            raw_values = data[feature]
            valid_values = Math._valid_values(raw_values)

            results[feature] = {
                "Count": Math.count(raw_values),
                "Mean": Math.mean(raw_values),
                "Std": Math.std(raw_values),
                "Variance": Math.variance(raw_values),
                "Min": Math.min(raw_values),
                "25%": Math.percentile(raw_values, 25),
                "50%": Math.percentile(raw_values, 50),
                "75%": Math.percentile(raw_values, 75),
                "Max": Math.max(raw_values),
                "Missing": len(raw_values) - len(valid_values),
                "Range": Math.range(raw_values),
                "Unique": Math.unique(raw_values),
            }

        results_table = pd.DataFrame(results)

        for row in ("Count", "Missing", "Unique"):
            results_table.loc[row] = (
                results_table.loc[row].astype(int)
            )

        print(results_table.to_string())

        return 0

    except FileNotFoundError:
        print("Error: file not found.")
        return 1

    except pd.errors.EmptyDataError:
        print("Error: the CSV file is empty.")
        return 1

    except (ValueError, KeyError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
