from calculs import(
    calculate_percentile,
    calculate_max,
    calculate_min,
    calculate_std,
    calculate_mean,
    calculate_count,
    calculate_correlation
)

import numpy  as np
import sys
import csv


def compare_homogeneous(data):

    houses = data["Hogwarts House"].dropna().unique()

    numeric_columns = data.select_dtypes(include=np.number).columns
    numeric_columns = numeric_columns.drop("Index")

    best_feature = None
    best_difference = None

    for feature in numeric_columns:

        means = []

        for house in houses:

            values = data[data["Hogwarts House"] == house][feature]
            values = values.dropna()

            mean = calculate_mean(values)
            means.append(mean)

        difference = calculate_max(means) - calculate_min(means)

        if best_difference is None or difference < best_difference:
            best_difference = difference
            best_feature = feature

    return best_feature

def compare_similar(data):
    
    numeric_columns = data.select_dtypes(include=np.number).columns
    numeric_columns = numeric_columns.drop("Index")
    
    best_feature_1 = None    
    best_feature_2 = None
    best_correlation = None
    
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            feature_1 = numeric_columns[i]
            feature_2 = numeric_columns[j]
            values = data[[feature_1, feature_2]].dropna()
            correlation = calculate_correlation(
                values[feature_1],
                values[feature_2]
            )

            correlation_absolute = abs(correlation)

            if (
                best_correlation is None
                or correlation_absolute > best_correlation
            ):
                best_correlation = correlation_absolute
                best_feature_1 = feature_1
                best_feature_2 = feature_2

    return best_feature_1, best_feature_2