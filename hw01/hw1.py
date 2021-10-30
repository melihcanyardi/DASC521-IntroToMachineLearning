import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generating Random Points
np.random.seed(421)

data_1 = np.random.multivariate_normal(mean=np.array([+0.0, +2.5]), cov=np.array([[+3.2, +0.0], [+0.0, +1.2]]), size=120)
data_2 = np.random.multivariate_normal(mean=np.array([-2.5, -2.0]), cov=np.array([[+1.2, +0.8], [+0.8, +1.2]]), size=80)
data_3 = np.random.multivariate_normal(mean=np.array([+2.5, -2.0]), cov=np.array([[+1.2, -0.8], [-0.8, +1.2]]), size=100)

X = np.vstack([data_1, data_2, data_3]) # data matrix
y = np.concatenate([np.repeat(1, data_1.shape[0]), np.repeat(2, data_2.shape[0]), np.repeat(3, data_3.shape[0])])


plt.figure(figsize = (10, 10))
plt.plot(data_1[:,0], data_1[:,1], 'ro')
plt.plot(data_2[:,0], data_2[:,1], 'go')
plt.plot(data_3[:,0], data_3[:,1], 'bo')
plt.xlabel("$x_1$", fontsize=15)
plt.ylabel("$x_2$", fontsize=15)
plt.xlim(-6, +6)
plt.ylim(-6, +6)
plt.show()


# Estimating the Parameters
## Sample Mean
def estimate_sample_mean(data):
    return [np.sum(data[:, i]) / data.shape[0] for i in range(data.shape[1])]

sample_means = np.array([estimate_sample_mean(data) for data in [data_1, data_2, data_3]])
print(sample_means)

## Sample Covariance Matrix
def estimate_covariance_matrix(data):
    return np.array([
        np.sum((data[:, 0] - estimate_sample_mean(data)[0]) * (data[:, 0] - estimate_sample_mean(data)[0])) / data.shape[0],
        np.sum((data[:, 0] - estimate_sample_mean(data)[0]) * (data[:, 1] - estimate_sample_mean(data)[1])) / data.shape[0],
        np.sum((data[:, 1] - estimate_sample_mean(data)[1]) * (data[:, 0] - estimate_sample_mean(data)[0])) / data.shape[0],
        np.sum((data[:, 1] - estimate_sample_mean(data)[1]) * (data[:, 1] - estimate_sample_mean(data)[1])) / data.shape[0]
    ]).reshape(2, 2)

sample_covariances = np.array([estimate_covariance_matrix(data) for data in [data_1, data_2, data_3]])
print(sample_covariances)

## Prior Probabilities
def prior_probability(data, X):
    return data.shape[0] / X.shape[0]

class_priors = [prior_probability(data, X) for data in [data_1, data_2, data_3]]
print(class_priors)


# Calculating Confusion Matrix
## Quadratic Discriminant
def quadratic_discriminant(x, sample_covariance, sample_mean, class_prior):
    return - (1/2 * np.log(np.linalg.det(sample_covariance))) - (1/2 * ((x - sample_mean) @ np.linalg.inv(sample_covariance) @ (x - sample_mean).T)) + (np.log(class_prior))

## Algorithm
def predict(x):
    g1 = quadratic_discriminant(x, sample_covariances[0], sample_means[0], class_priors[0])
    g2 = quadratic_discriminant(x, sample_covariances[1], sample_means[1], class_priors[1])
    g3 = quadratic_discriminant(x, sample_covariances[2], sample_means[2], class_priors[2])
    
    if g1 > g2 and g1 > g3:
        return 1
    elif g2 > g1 and g2 > g3:
        return 2
    else:
        return 3

y_pred = np.array([predict(x) for x in X])

## Confusion Matrix
confusion_matrix = pd.crosstab(y_pred, y, rownames = ["y_pred"], colnames = ["y_truth"])
print(confusion_matrix)


# Drawing Decision Boundaries
## Quadratic Discriminant
def Wi(sample_covariance):
    return -1/2 * np.linalg.inv(sample_covariance)

W1, W2, W3 = [Wi(sample_covariance) for sample_covariance in sample_covariances]

print(f"W1:\n{W1}\n")
print(f"W2:\n{W2}\n")
print(f"W3:\n{W3}")

def wi(sample_covariance, sample_mean):
    return np.linalg.inv(sample_covariance) @ sample_mean

w1, w2, w3 = [wi(sample_covariance, sample_mean) for sample_covariance, sample_mean in zip(sample_covariances, sample_means)]

print(f"w1:\n{w1}\n")
print(f"w2:\n{w2}\n")
print(f"w3:\n{w3}")

def wi0(sample_covariance, sample_mean, class_prior):
    return - (1/2 * (sample_mean.T @ np.linalg.inv(sample_covariance) @ sample_mean)) - (1/2 * np.log(np.linalg.det(sample_covariance))) + np.log(class_prior)

w10, w20, w30 = [wi0(sample_covariance, sample_mean, class_prior) for sample_covariance, sample_mean, class_prior in zip(sample_covariances, sample_means, class_priors)]

print(f"w10:\n{w10}\n")
print(f"w20:\n{w20}\n")
print(f"w30:\n{w30}")



x1_interval = np.linspace(-6, +6, 1200)
x2_interval = np.linspace(-6, +6, 1200)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)

discriminant_values_1 = (x1_grid * W1[0, 0] * x1_grid.T) + (x1_grid * W1[0, 1] * x1_grid.T) + (x2_grid * W1[1, 0] * x2_grid.T) + (x2_grid * W1[1, 1] * x2_grid.T) + (w1[0] * x1_grid) + (w1[1] * x2_grid) + w10
discriminant_values_2 = (x1_grid * W2[0, 0] * x1_grid.T) + (x1_grid * W2[0, 1] * x1_grid.T) + (x2_grid * W2[1, 0] * x2_grid.T) + (x2_grid * W2[1, 1] * x2_grid.T) + (w2[0] * x1_grid) + (w2[1] * x2_grid) + w20
discriminant_values_3 = (x1_grid * W3[0, 0] * x1_grid.T) + (x1_grid * W3[0, 1] * x1_grid.T) + (x2_grid * W3[1, 0] * x2_grid.T) + (x2_grid * W3[1, 1] * x2_grid.T) + (w3[0] * x1_grid) + (w3[1] * x2_grid) + w30

plt.figure(figsize = (10, 10))
plt.plot(X[y == 1, 0], X[y == 1, 1], "ro")
plt.plot(X[y == 2, 0], X[y == 2, 1], "bo")
plt.plot(X[y == 3, 0], X[y == 3, 1], "go")
plt.plot(X[y_pred != y, 0], X[y_pred != y, 1], "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values_1, levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values_2, levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values_3, levels = 0, colors = "k")
plt.xlabel("$x_1$", fontsize=15)
plt.ylabel("$x_2$", fontsize=15)

plt.show()

### Linear Decision Boundaries
discriminant_values_1 = (w1[0] * x1_grid) + (w1[1] * x2_grid) + w10
discriminant_values_2 = (w2[0] * x1_grid) + (w2[1] * x2_grid) + w20
discriminant_values_3 = (w3[0] * x1_grid) + (w3[1] * x2_grid) + w30

plt.figure(figsize = (10, 10))
plt.plot(X[y == 1, 0], X[y == 1, 1], "ro")
plt.plot(X[y == 2, 0], X[y == 2, 1], "bo")
plt.plot(X[y == 3, 0], X[y == 3, 1], "go")
plt.plot(X[y_pred != y, 0], X[y_pred != y, 1], "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values_1, levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values_2, levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values_3, levels = 0, colors = "k")
plt.xlabel("$x_1$", fontsize=15)
plt.ylabel("$x_2$", fontsize=15)

plt.show()

