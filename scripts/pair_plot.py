import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide a CSV file.")

        filename = sys.argv[1]

        if not filename.lower().endswith(".csv"):
            raise ValueError("The file must be a CSV file.")

        data = pd.read_csv(filename)

        print("CSV file successfully loaded.")

        groups = [
            [
                "Arithmancy",
                "Astronomy",
                "Defense Against the Dark Arts",
            ],
            [
                "Divination",
                "Muggle Studies",
                "History of Magic",
            ],
            [
                "Transfiguration",
                "Care of Magical Creatures",
                "Charms",
                "Flying",
            ],
            [
                "Herbology",
                "Potions",
                "Ancient Runes",
                "Charms",
            ],
        ]

        if "Hogwarts House" not in data.columns:
            raise ValueError("Missing 'Hogwarts House' column.")

        for group in groups:
            missing = [
                feature
                for feature in group
                if feature not in data.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing features: {', '.join(missing)}"
                )

            pair_data = data[
                ["Hogwarts House"] + group
            ].dropna()

            if pair_data.empty:
                print(f"No usable data for group: {group}")
                continue

            sns.pairplot(
                pair_data,
                hue="Hogwarts House"
            )

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