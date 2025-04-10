import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# (a) Load the dataset
data = pd.read_csv("winequality-red.csv", sep=";")

# (b) Split into 80% train and 20% test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare inputs
X_train = train_data.drop(columns=['quality']).values
y_train = train_data['quality'].values
X_test = test_data.drop(columns=['quality']).values
y_test = test_data['quality'].values

# Add bias term (intercept)
X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# (c) Closed-form solution for linear regression
# Done using numpy's matrix multiplication.
def closed_form_solution(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

w_closed = closed_form_solution(X_train_bias, y_train)

# Predictions manually computed.
y_train_pred = X_train_bias @ w_closed
y_test_pred = X_test_bias @ w_closed

# (d) Plot actual vs predicted values
plt.scatter(y_train, y_train_pred, alpha=0.6)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Actual vs Predicted (Train Set)")
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.grid(True)
plt.show()
# Interpretation: The closer points are to the red line, the more accurate the predictions. 

# (e) Root Mean Square Error
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

print(f"(e) RMSE on Train Set: {rmse(y_train, y_train_pred):.4f}")
print(f"    RMSE on Test Set: {rmse(y_test, y_test_pred):.4f}")

# (f) Split train into smaller train and validation (75%-25%)
X_subtrain, X_val, y_subtrain, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
X_subtrain_bias = np.hstack([np.ones((X_subtrain.shape[0], 1)), X_subtrain])
X_val_bias = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

# (g) Least-Mean-Squares (LMS) / Gradient Descent
def lms(X, y, w0, eta, nums=100):
    w = w0.copy()
    for num in range(nums):
        for i in range(X.shape[0]):
            xi = X[i].reshape(1, -1)
            yi = y[i]
            gradient = 2 * xi.T @ (xi @ w - yi)
            w -= eta * gradient
    return w

# (h) Tune step size eta, done by trial and error. Settled on one that converged in the smallest number of iterations. Learned in another class to try step sizes on different exponential orders since 0.01 and 0.02 for example are likely to behave similarly.
# w0 is initialized from a normal distribution with mean 0 and standard deviation 0.01. 
etas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
errors = []
np.random.seed(42)
w0 = np.random.normal(0, 0.01, size=X_subtrain_bias.shape[1])

for eta in etas:
    w_eta = lms(X_subtrain_bias, y_subtrain, w0, eta)
    y_val_pred = X_val_bias @ w_eta
    error = rmse(y_val, y_val_pred)
    errors.append(error)
    print(f"(h) RMSE with η={eta:.0e}: {error:.4f}")

best_eta = etas[np.argmin(errors)]
print(f"    Best η: {best_eta}")

# (i) Train LMS on full train set with best eta
w_final = lms(X_train_bias, y_train, w0, best_eta)

# (j) Final RMSE on train and test
final_train_pred = X_train_bias @ w_final
final_test_pred = X_test_bias @ w_final
print(f"(j) Final LMS RMSE on Train Set: {rmse(y_train, final_train_pred):.4f}")
print(f"    Final LMS RMSE on Test Set: {rmse(y_test, final_test_pred):.4f}")
