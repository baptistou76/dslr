import pandas as pd
import matplotlib.pyplot as plt
import sys

from DSLR.comparate import compare_homogeneous


def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide a CSV file.")

        filename = sys.argv[1]

        if not filename.lower().endswith(".csv"):
            raise ValueError("The file must be a CSV file.")

        data = pd.read_csv(filename)

        print("CSV file successfully loaded.")

        feature = compare_homogeneous(data)

        print(f"Most homogeneous feature: {feature}")

        houses = data["Hogwarts House"].dropna().unique()

        plt.figure(figsize=(10, 6))

        for house in houses:
            values = data.loc[
                data["Hogwarts House"] == house,
                feature
            ].dropna()

            plt.hist(
                values,
                bins=30,
                alpha=0.5,
                label=house
            )

        plt.title(feature)
        plt.xlabel("Score")
        plt.ylabel("Number of students")
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