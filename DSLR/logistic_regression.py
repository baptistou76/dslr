
import numpy as np


class LogisticRegression:
    """Binary logistic regression trained with gradient descent."""

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.mean = None
        self.std = None

    @staticmethod
    def sigmoid(z):
        """Convert scores into probabilities."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Train the model on a binary classification dataset."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        if len(X) == 0:
            raise ValueError("Cannot train on an empty dataset.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values.")

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only 0 and 1.")

        if self.iterations <= 0 or self.learning_rate <= 0:
            raise ValueError("Learning rate and iterations must be positive.")

        # Normalize each feature.
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1

        X_scaled = (X - self.mean) / self.std

        # Add the bias column.
        X_bias = np.column_stack((np.ones(len(X_scaled)), X_scaled))

        # Initialize weights.
        self.weights = np.zeros(X_bias.shape[1])

        # Gradient descent.
        for _ in range(self.iterations):
            probabilities = self.sigmoid(X_bias @ self.weights)
            gradient = X_bias.T @ (probabilities - y) / len(y)
            self.weights -= self.learning_rate * gradient

        return self

    def predict_proba(self, X):
        """Return the probability of class 1."""
        if self.weights is None:
            raise ValueError("The model must be trained before prediction.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[1] != len(self.mean):
            raise ValueError("X does not have the expected number of features.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        X_scaled = (X - self.mean) / self.std
        X_bias = np.column_stack((np.ones(len(X_scaled)), X_scaled))

        return self.sigmoid(X_bias @ self.weights)

    def predict(self, X, threshold=0.5):
        """Predict binary classes: 0 or 1."""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
