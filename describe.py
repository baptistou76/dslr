import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

from calculs import (
    calculate_percentile,
    calculate_count,
    calculate_min,
    calculate_max,
    calculate_mean,
    calculate_std
)


def main():

    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide a CSV file.")

        filename = sys.argv[1]

        if not filename.lower().endswith(".csv"):
            raise ValueError("The file must be a CSV file.")

        data = pd.read_csv(filename)

        print("CSV file successfully loaded.")

        numeric_columns = data.select_dtypes(include=np.number).columns
        numeric_columns = numeric_columns.drop("Index")

        results = {}

        for features in numeric_columns:

            values = data[features]

            count = calculate_count(values)
            mean = calculate_mean(values)
            std = calculate_std(values)
            minimum = calculate_min(values)
            maximum = calculate_max(values)

            percentile_25 = calculate_percentile(values, 25)
            percentile_50 = calculate_percentile(values, 50)
            percentile_75 = calculate_percentile(values, 75)

            results[features] = {
                "Count": count,
                "Mean": mean,
                "Std": std,
                "Min": minimum,
                "25%": percentile_25,
                "50%": percentile_50,
                "75%": percentile_75,
                "Max": maximum
            }

        results_table = pd.DataFrame(results)

        results_table.loc["Count"] = results_table.loc["Count"].astype(int)

        print(results_table)

    except FileNotFoundError:
        print("Error: file not found.")

    except pd.errors.EmptyDataError:
        print("Error: the CSV file is empty.")

    except ValueError as error:
        print(f"Error: {error}")

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()