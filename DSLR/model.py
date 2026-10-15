
import numpy as np

from .logistic_regression import LogisticRegression


class Model:
    """Manage the four binary classifiers for Hogwarts houses."""

    HOUSES = [
        "Gryffindor",
        "Hufflepuff",
        "Ravenclaw",
        "Slytherin",
    ]

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.classifiers = {}
        self.features = None
        self.imputation_means = None

    def fit(self, X, y):
        """Train one binary classifier per house."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        if len(X) == 0:
            raise ValueError("Cannot train on an empty dataset.")

        if not np.all(np.isin(y, self.HOUSES)):
            raise ValueError("y contains an unknown house.")

        self.classifiers = {}

        for house in self.HOUSES:
            print(f"Training classifier: {house}")

            # 1 = current house, 0 = all other houses.
            y_binary = (y == house).astype(int)

            classifier = LogisticRegression(
                learning_rate=self.learning_rate,
                iterations=self.iterations,
            )

            classifier.fit(X, y_binary)
            self.classifiers[house] = classifier

        return self

    def predict_proba(self, X):
        """Return one probability per house and sample."""
        if len(self.classifiers) != len(self.HOUSES):
            raise ValueError("The model must be trained before prediction.")

        probabilities = []

        for house in self.HOUSES:
            house_probabilities = self.classifiers[house].predict_proba(X)
            probabilities.append(house_probabilities)

        return np.column_stack(probabilities)

    def predict(self, X):
        """Predict the most likely house for each sample."""
        probabilities = self.predict_proba(X)
        best_indices = np.argmax(probabilities, axis=1)

        return np.array(self.HOUSES)[best_indices]

    def save(self, filepath="logreg_model.npz"):
        """Save all trained classifiers and their normalization parameters."""
        if len(self.classifiers) != len(self.HOUSES):
            raise ValueError("The model must be trained before saving.")

        data = {}
        if self.features is not None:
                    data["features"] = np.asarray(self.features, dtype=str)
                    
        if self.imputation_means is not None:
            data["imputation_means"] = np.asarray(
                    self.imputation_means,
                    dtype=float,
                )

        for index, house in enumerate(self.HOUSES):
            classifier = self.classifiers[house]

            data[f"weights_{index}"] = classifier.weights
            data[f"mean_{index}"] = classifier.mean
            data[f"std_{index}"] = classifier.std

        np.savez(filepath, **data)
        print(f"Model saved to {filepath}")

    def load(self, filepath="logreg_model.npz"):
        """Load the trained classifiers and their normalization parameters."""
        with np.load(filepath) as data:
            classifiers = {}

            for index, house in enumerate(self.HOUSES):
                classifier = LogisticRegression(
                    learning_rate=self.learning_rate,
                    iterations=self.iterations,
                )
                
                self.features = (
                    data["features"].astype(str).tolist()
                    if "features" in data.files
                    else None
                )
                
                self.imputation_means = (
                    data["imputation_means"].copy()
                    if "imputation_means" in data.files
                    else None
                )

                classifier.weights = data[f"weights_{index}"].copy()
                classifier.mean = data[f"mean_{index}"].copy()
                classifier.std = data[f"std_{index}"].copy()

                classifiers[house] = classifier

        self.classifiers = classifiers
        print(f"Model loaded from {filepath}")

        return self
