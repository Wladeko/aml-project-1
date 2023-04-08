import numpy as np
from rich import print
from rich.table import Table


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_regression_irls(X, y, interaction_pairs=None, tol=1e-6, max_iter=100, delta=1e-4):
    n, p = X.shape

    # Add a column of ones for the intercept
    X = np.hstack([np.ones((n, 1)), X])

    # Add interactions
    if interaction_pairs is not None:
        X_interact = []
        for i, j in interaction_pairs:
            X_interact.append(X[:, i] * X[:, j])
        X_interact = np.array(X_interact).T
        X = np.hstack([X, X_interact])

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


def generate_artificial_data(num_samples, num_features, interaction_pairs):
    # generate original feature matrix
    X = np.random.normal(size=(num_samples, num_features))

    # generate interaction features
    interaction_features = []
    for i, j in interaction_pairs:
        interaction_features.append(X[:, i] * X[:, j])
    interaction_features = np.array(interaction_features).T

    # concatenate original and interaction features
    X_extended = np.hstack([np.ones((num_samples, 1)), X, interaction_features])

    # generate true weight vector
    true_weights = np.random.normal(size=(num_features + 1 + len(interaction_pairs), 1))

    # generate target vector
    logits = X_extended @ true_weights
    probabilities = sigmoid(logits)
    y = np.random.binomial(1, probabilities).reshape(-1, 1)

    return X, X_extended, y, true_weights


def main():
    np.random.seed(123)

    # hyperparameters
    num_samples = 2000
    num_features = 5
    interaction_pairs = [(1, 3), (2, 4)]

    X, X_extended, y, w_true = generate_artificial_data(
        num_samples=num_samples,
        num_features=num_features,
        interaction_pairs=interaction_pairs,
    )

    w = logistic_regression_irls(X, y, interaction_pairs=interaction_pairs)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Coefficients")
    table.add_column("True")
    table.add_column("Estimated")
    table.add_column("MAE", style="bold green")

    for i in range(num_features):
        table.add_row(
            f"Feature {i+1}",
            f"{float(w_true[i]):.3f}",
            f"{float(w[i]):.3f}",
            f"{float(abs(w[i] - w_true[i])):.3f}",
        )

    print(table)

    mae = np.mean(np.abs(w - w_true))
    print("Final MAE:", mae)


if __name__ == "__main__":
    main()
