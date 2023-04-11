import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_data(num_samples, num_features, interaction_pairs=None):
    # generate original feature matrix
    X = np.random.normal(size=(num_samples, num_features))

    # generate interaction features
    interaction_features = None
    if interaction_pairs:
        interaction_features = [X[:, i] * X[:, j] for i, j in interaction_pairs]
        interaction_features = np.array(interaction_features).T

    # concatenate original and interaction features
    if interaction_features is not None:
        X_extended = np.hstack([np.ones((num_samples, 1)), X, interaction_features])
    else:
        X_extended = np.hstack([np.ones((num_samples, 1)), X])

    # generate true weight vector
    num_interaction_features = 0 if not interaction_pairs else len(interaction_pairs)
    true_weights = np.random.normal(size=(1 + num_features + num_interaction_features, 1))

    # generate target vector
    logits = X_extended @ true_weights
    probabilities = sigmoid(logits)
    y = np.random.binomial(1, probabilities).reshape(-1, 1)

    # Ravel
    y = y.ravel()

    return X, y, true_weights
