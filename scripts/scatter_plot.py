import pandas as pd
import matplotlib.pyplot as plt
import sys

from DSLR.comparate import compare_similar


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
            raise ValueError("Missing 'Hogwarts House' column.")

        feature_1, feature_2 = compare_similar(data)

        print(f"Similar features: {feature_1} and {feature_2}")

        values = data[
            ["Hogwarts House", feature_1, feature_2]
        ].dropna()

        plt.figure(figsize=(10, 6))

        for house in values["Hogwarts House"].unique():
            house_values = values[
                values["Hogwarts House"] == house
            ]

            plt.scatter(
                house_values[feature_1],
                house_values[feature_2],
                alpha=0.5,
                label=house
            )

        plt.title("Most correlated features")
        plt.xlabel(feature_1)
        plt.ylabel(feature_2)
        plt.legend(title="Hogwarts House")
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