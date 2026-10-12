import pandas as pd
import numpy as np
from calculs import calculate_mean, calculate_std
import sys

FEATURES = [
    "Herbology",
    "Defense Against the Dark Arts",
    "Muggle Studies",
    "History of Magic",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying"
]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):

    number_of_samples = X.shape[0]
    number_of_features = X.shape[1]

    weights = np.zeros(number_of_features)
    bias = 0

    for _ in range(iterations):

        z = np.dot(X, weights) + bias

        predictions = sigmoid(z)

        error = predictions - y

        gradient_weights = np.dot(X.T, error) / number_of_samples
        gradient_bias = np.sum(error) / number_of_samples

        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    return weights, bias

def main():

    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide a CSV file.")

        filename = sys.argv[1]

        if not filename.lower().endswith(".csv"):
            raise ValueError("The file must be a CSV file.")

        data = pd.read_csv(filename)

        print("CSV file successfully loaded.")

        data = data.dropna(subset=FEATURES + ["Hogwarts House"])

        X = data[FEATURES].values
        y = (data["Hogwarts House"] == "Ravenclaw").astype(int).values

        means = np.calculate_mean(X, axis=0)
        stds = np.calculate_std(X, axis=0)

        X = (X - means) / stds

        weights, bias = gradient_descent(X, y)

        print("Training finished.")
        print("Weights:")
        print(weights)
        print("Bias:")
        print(bias)

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
