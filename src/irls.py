import numpy as np
from rich import print
from rich.table import Table

from src.data import artificial


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_regression_irls(X, y, interaction_pairs=None, tol=1e-6, max_iter=100):
    # generate interaction features
    if interaction_pairs:
        interaction_features = [X[:, i] * X[:, j] for i, j in interaction_pairs]
        interaction_features = np.array(interaction_features).T
        X = np.hstack([X, interaction_features])

    # Add a column of ones for the intercept
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    n, p = X.shape

    # Initialize the coefficients to zero
    w = np.zeros((p, 1))

    for _ in range(max_iter):
        # Compute the predicted probabilities
        p = sigmoid(X @ w)

        # Compute the diagonal weight matrix
        W = np.diagflat(p * (1 - p))

        # Compute the Hessian
        H = X.T @ W @ X
        H_inv = np.linalg.inv(H)

        # Update the coefficients using Newton's method
        z = X @ w + np.linalg.inv(W) @ (y - p)
        w_new = H_inv @ X.T @ W @ z

        # Check for convergence
        if np.linalg.norm(w_new - w) < tol:
            break

        w = w_new

    return w


def main():
    np.random.seed(123)

    # hyperparameters
    num_samples = 2000
    num_features = 5
    interaction_pairs = [(1, 3), (2, 4)]

    X, y, true_weights = artificial.generate_data(
        num_samples=num_samples,
        num_features=num_features,
        interaction_pairs=interaction_pairs,
    )

    weights = logistic_regression_irls(X, y, interaction_pairs=interaction_pairs)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Coefficients")
    table.add_column("True")
    table.add_column("Estimated")
    table.add_column("MAE", style="bold green")

    for i in range(num_features):
        table.add_row(
            f"Feature {i+1}",
            f"{float(true_weights[i]):.3f}",
            f"{float(weights[i]):.3f}",
            f"{float(abs(weights[i] - true_weights[i])):.3f}",
        )

    print(table)

    mae = np.mean(np.abs(weights - true_weights))
    print("Final MAE:", mae)


if __name__ == "__main__":
    main()
