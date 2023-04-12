import numpy as np
import pandas as pd


def prepare_data():
    # Load the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    df = pd.read_csv(url, header=None)

    # Rename columns
    df.columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    # Replace missing values "?" with median
    df.replace("?", np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Convert categorical variables into numerical ones
    cat_vars = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
    df = pd.get_dummies(df, columns=cat_vars)

    # Remove collinear variables
    corr_matrix = df.corr(numeric_only=False).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.5) and column != "target"]
    df.drop(to_drop, axis=1, inplace=True)
    print("Dropped colinear columns:", to_drop)

    # Map target variable to binary labels
    df["target"] = df["target"].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

    X = df.drop("target", axis=1)
    y = df["target"]

    # To numpy float
    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(np.int64)

    # Ravel
    y = y.ravel()

    return X, y
