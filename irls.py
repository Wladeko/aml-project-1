import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_regression_irls(X, y, interactions=None, tol=1e-6, max_iter=100, delta=1e-4):
    n, p = X.shape
    # Add a column of ones for the intercept
    X = np.hstack([np.ones((n, 1)), X])
    
    # Add interactions
    if interactions is not None:
        X_interact = []
        for i, j in interactions:
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
        5
        # Check for convergence
        if np.linalg.norm(w_new - w) < tol:
            break
        
        w = w_new
    
    return w


# Test
# np.random.seed(123)

m = 2000
n = 5

X = np.random.normal(size=(m, n))

interactions = [(1, 3), (2, 4)]
X_interact = []
for i, j in interactions:
    X_interact.append(X[:, i] * X[:, j])
X_interact = np.array(X_interact).T
X_extendend = np.hstack([np.ones((m, 1)), X, X_interact])

w_true = np.random.normal(size=(n+1+len(interactions), 1))  # include intercept and interactions

# Generate the target vector y using logistic regression
p = sigmoid(X_extendend @ w_true)
y = np.random.binomial(1, p).reshape(-1, 1)

w = logistic_regression_irls(X, y, interactions=interactions)


# check
# Print the true and estimated coefficients
print("True coefficients:\n", w_true)
print("Estimated coefficients:\n", w)

# Compute the mean absolute error (MAE)
mae = np.mean(np.abs(w - w_true))
print("MAE:", mae)