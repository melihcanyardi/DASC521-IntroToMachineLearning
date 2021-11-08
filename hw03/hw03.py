import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Generating Random Points
np.random.seed(421)

data_1 = np.random.multivariate_normal(mean=np.array([+0.0, +2.5]),
                                       cov=np.array([[+3.2, +0.0],
                                                     [+0.0, +1.2]]), size=120)
data_2 = np.random.multivariate_normal(mean=np.array([-2.5, -2.0]),
                                       cov=np.array([[+1.2, +0.8],
                                                     [+0.8, +1.2]]), size=80)
data_3 = np.random.multivariate_normal(mean=np.array([+2.5, -2.0]),
                                       cov=np.array([[+1.2, -0.8],
                                                     [-0.8, +1.2]]), size=100)

X = np.vstack([data_1, data_2, data_3]) # data matrix
y_truth = np.concatenate([np.repeat(1, data_1.shape[0]),
                          np.repeat(2, data_2.shape[0]),
                          np.repeat(3, data_3.shape[0])])

data_set = np.hstack((X, y_truth[:, None]))

print(f"Data matrix:\n{X}\n")
print(f"y_truth:\n{y_truth}\n")
print(f"Data set:\n{data_set}\n")


# Plotting Data
plt.figure(figsize = (10, 10))
plt.plot(data_1[:,0], data_1[:,1], 'ro')
plt.plot(data_2[:,0], data_2[:,1], 'go')
plt.plot(data_3[:,0], data_3[:,1], 'bo')
plt.xlabel("$x_1$", fontsize=15)
plt.ylabel("$x_2$", fontsize=15)
plt.xlim(-6, +6)
plt.ylim(-6, +6)
plt.show()


# Number of classes, number of samples, and one-of-K encoding
## K: number of classes
K = np.max(y_truth)

## N: number of samples
N = data_set.shape[0]

## one-of-K encoding
Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth - 1] = 1

print(f"K, N: {(K, N)}\n")

print(f"Y_truth (one-of-K encoding):\n{Y_truth}\n")


# Sigmoid Function
def sigmoid(X, W, w0):
    return 1 / (1 + np.exp(-(X@W) + np.repeat(w0, X.shape[0], axis=0)))


# Gradient Function
def gradient_W(X, Y_truth, Y_predicted):
    return(np.asarray([-np.matmul(Y_truth[:,c] - Y_predicted[:,c], X) for c in range(K)]).transpose())

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum(Y_truth - Y_predicted, axis = 0))


# Algorithm Parameters
eta = 0.01
epsilon = 0.001


# Parameter Initialization
np.random.seed(421)

W = np.random.uniform(low = -0.01, high = 0.01, size = (X.shape[1], K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))


# Iterative Algorithm
iteration = 1
objective_values = []

print("Iterative Algorithm")
while 1:
    print(f"iteration #{iteration}")
    Y_predicted = sigmoid(X, W, w0)

    objective_values = np.append(objective_values, 0.5*np.sum((Y_truth - Y_predicted)**2))

    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(X, Y_truth, Y_predicted)
    w0 = w0 + eta * gradient_w0(Y_truth, Y_predicted)

    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break

    iteration = iteration + 1

print(f"\nIteration completed in {iteration} steps.\n")

# Parameter Estimations
print(f"W:\n{W}\n")
print(f"w0:\n{w0}\n")


# Convergence
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# Confusion Matrix
y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(y_predicted, y_truth,
                               rownames = ['y_pred'], colnames = ['y_truth'])

print(f"Confusion Matrix:\n\n{confusion_matrix}\n")


# Drawing Decision Boundaries
x1_interval = np.linspace(-8, +8, 1201)
x2_interval = np.linspace(-8, +8, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))
for c in range(K):
    discriminant_values[:,:,c] = W[0, c] * x1_grid + W[1, c] * x2_grid + w0[0, c]

A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:,:,0] = A
discriminant_values[:,:,1] = B
discriminant_values[:,:,2] = C

plt.figure(figsize = (10, 10))
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "r.", markersize = 10)
plt.plot(X[y_truth == 2, 0], X[y_truth == 2, 1], "g.", markersize = 10)
plt.plot(X[y_truth == 3, 0], X[y_truth == 3, 1], "b.", markersize = 10)
plt.plot(X[y_predicted != y_truth, 0], X[y_predicted != y_truth, 1], "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,1], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()