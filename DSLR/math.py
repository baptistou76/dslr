
import math


class Math:
    """Mathematical and statistical utility functions."""

    @staticmethod
    def _valid_values(values):
        """Return valid finite numeric values."""
        result = []

        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(number):
                result.append(number)

        return result

    @staticmethod
    def count(values):
        return len(Math._valid_values(values))

    @staticmethod
    def mean(values):
        values = Math._valid_values(values)

        if not values:
            raise ValueError("Cannot calculate mean: no valid values.")

        return sum(values) / len(values)

    @staticmethod
    def std(values):
        values = Math._valid_values(values)

        if not values:
            raise ValueError(
                "Cannot calculate standard deviation: no valid values."
            )

        average = Math.mean(values)
        squared_differences = [
            (value - average) ** 2 for value in values
        ]

        variance = sum(squared_differences) / len(values)

        return math.sqrt(variance)

    @staticmethod
    def min(values):
        values = Math._valid_values(values)

        if not values:
            raise ValueError("Cannot calculate minimum: no valid values.")

        smallest = values[0]

        for value in values[1:]:
            if value < smallest:
                smallest = value

        return smallest

    @staticmethod
    def max(values):
        values = Math._valid_values(values)

        if not values:
            raise ValueError("Cannot calculate maximum: no valid values.")

        largest = values[0]

        for value in values[1:]:
            if value > largest:
                largest = value

        return largest

    @staticmethod
    def percentile(values, percentile):
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100.")

        values = Math._valid_values(values)

        if not values:
            raise ValueError(
                "Cannot calculate percentile: no valid values."
            )

        values = sorted(values)

        position = (len(values) - 1) * percentile / 100
        lower = int(position)
        upper = lower + 1

        if upper >= len(values):
            return values[lower]

        weight = position - lower

        return (
            values[lower] * (1 - weight)
            + values[upper] * weight
        )
        
    @staticmethod
    def correlation(values_x, values_y):
        pairs = []

        for x, y in zip(values_x, values_y):
            try:
                x = float(x)
                y = float(y)
            except (TypeError, ValueError):
                continue

            if math.isfinite(x) and math.isfinite(y):
                pairs.append((x, y))

        if len(pairs) < 2:
            return 0.0

        valid_x = [x for x, _ in pairs]
        valid_y = [y for _, y in pairs]

        mean_x = Math.mean(valid_x)
        mean_y = Math.mean(valid_y)

        numerator = sum(
            (x - mean_x) * (y - mean_y)
            for x, y in pairs
        )

        total_x = sum((x - mean_x) ** 2 for x in valid_x)
        total_y = sum((y - mean_y) ** 2 for y in valid_y)

        denominator = math.sqrt(total_x * total_y)

        if denominator == 0:
            return 0.0

        return numerator / denominator
    
    @staticmethod
    def variance(values):
        """Calculate population variance."""
        valid_values = Math._valid_values(values)
        if not valid_values:
            raise ValueError(
                "Cannot calculate variance: no valid values."
            )
        average = Math.mean(valid_values)
        squared_differences = [
            (value - average) ** 2
            for value in valid_values
        ]
        return sum(squared_differences) / len(valid_values)
    
    @staticmethod
    def range(values):
        """Calculate the range between maximum and minimum"""
        return Math.max(values) - Math.min(values)
    
    @staticmethod
    def unique(values):
        """Count distinct valid numeric values."""
        return len(set(Math._valid_values(values)))