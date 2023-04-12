import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class IRLS:
    def __init__(self, interaction_pairs=None, tol=1e-6, max_iter=100):
        self.interaction_pairs = interaction_pairs
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        # unravel
        y = y.reshape(-1, 1)

        # generate interaction features
        if self.interaction_pairs:
            interaction_features = [X[:, i] * X[:, j] for i, j in self.interaction_pairs]
            interaction_features = np.array(interaction_features).T
            X = np.hstack([X, interaction_features])

        # Add a column of ones for the intercept
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        n, p = X.shape

        # Initialize the coefficients to zero
        w = np.zeros((p, 1))

        for _ in range(self.max_iter):
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
            if np.linalg.norm(w_new - w) < self.tol:
                break

            w = w_new

        self.coef_ = w

        return w

    def predict(self, X):
        # generate interaction features
        if self.interaction_pairs:
            interaction_features = [X[:, i] * X[:, j] for i, j in self.interaction_pairs]
            interaction_features = np.array(interaction_features).T
            X = np.hstack([X, interaction_features])

        # Add a column of ones for the intercept
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Predict using the coefficients
        y_pred = sigmoid(X @ self.coef_)

        return y_pred
