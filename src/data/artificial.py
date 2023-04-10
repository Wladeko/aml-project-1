import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_data(num_samples, num_features, interaction_pairs=None):
    # generate original feature matrix
    X = np.random.normal(size=(num_samples, num_features))

    # generate interaction features
    if interaction_pairs:
        interaction_features = [X[:, i] * X[:, j] for i, j in interaction_pairs]
        interaction_features = np.array(interaction_features).T
        X_extended = np.hstack([X, interaction_features])

    # concatenate original and interaction features
    X_extended = np.hstack([np.ones((num_samples, 1)), X_extended])

    # generate true weight vector
    true_weights = np.random.normal(size=(1 + num_features + len(interaction_pairs), 1))

    # generate target vector
    logits = X_extended @ true_weights
    probabilities = sigmoid(logits)
    y = np.random.binomial(1, probabilities).reshape(-1, 1)

    return X, y, true_weights
