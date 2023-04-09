import numpy as np
import pandas as pd


def prepare_data():
    # Read the Titanic dataset
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # Convert categorical columns to one-hot encoding
    categorical_cols = ["Sex", "Embarked", "Pclass"]
    df = pd.get_dummies(df, columns=categorical_cols)

    # Replace missing values with median
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Cabin"].fillna("Unknown", inplace=True)

    # Drop irrelevant columns
    df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

    # Remove collinear variables
    # corr_matrix = df.corr().abs()
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    # df = df.drop(to_drop, axis=1)
    # print("Dropped:", to_drop)

    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]

    return X, y
