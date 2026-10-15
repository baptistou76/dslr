import pandas as pd

from DSLR.math import Math


def get_numeric_features(data):
    numeric_columns = data.select_dtypes(include="number").columns

    return [
        column
        for column in numeric_columns
        if column != "Index"
    ]


def compare_homogeneous(data):
    if "Hogwarts House" not in data.columns:
        raise ValueError("Missing 'Hogwarts House' column.")

    houses = data["Hogwarts House"].dropna().unique()
    features = get_numeric_features(data)

    if len(houses) == 0:
        raise ValueError("No Hogwarts House values found.")

    if len(features) == 0:
        raise ValueError("No numeric features found.")

    best_feature = None
    best_difference = None

    for feature in features:
        means = []

        for house in houses:
            values = data.loc[
                data["Hogwarts House"] == house,
                feature
            ].dropna()

            if values.empty:
                break

            means.append(Math.mean(values))

        if len(means) != len(houses):
            continue

        difference = Math.max(means) - Math.min(means)

        if (
            best_difference is None
            or difference < best_difference
        ):
            best_difference = difference
            best_feature = feature

    if best_feature is None:
        raise ValueError("Unable to find a homogeneous feature.")

    return best_feature


def compare_similar(data):
    features = get_numeric_features(data)

    if len(features) < 2:
        raise ValueError("At least two numeric features are required.")

    best_feature_1 = None
    best_feature_2 = None
    best_correlation = -1.0

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature_1 = features[i]
            feature_2 = features[j]

            values = data[[feature_1, feature_2]].dropna()

            correlation = Math.correlation(
                values[feature_1],
                values[feature_2]
            )

            correlation_absolute = abs(correlation)

            if correlation_absolute > best_correlation:
                best_correlation = correlation_absolute
                best_feature_1 = feature_1
                best_feature_2 = feature_2

    if best_feature_1 is None:
        raise ValueError("Unable to find similar features.")

    return best_feature_1, best_feature_2