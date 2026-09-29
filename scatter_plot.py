import pandas as pd
import matplotlib.pyplot as plt
import sys

from comparate import compare_similar


def main():

    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide a CSV file.")

        filename = sys.argv[1]

        if not filename.lower().endswith(".csv"):
            raise ValueError("The file must be a CSV file.")

        data = pd.read_csv(filename)

        print("CSV file successfully loaded.")

        feature_1, feature_2 = compare_similar(data)

        print(f"Similar features: {feature_1} and {feature_2}")

        values = data[[feature_1, feature_2]].dropna()

        plt.figure(figsize=(10, 6))

        plt.scatter(
            values[feature_1],
            values[feature_2],
            alpha=0.5
        )

        plt.title("Similar features")
        plt.xlabel(feature_1)
        plt.ylabel(feature_2)
        plt.legend()

        plt.tight_layout()
        plt.show()

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