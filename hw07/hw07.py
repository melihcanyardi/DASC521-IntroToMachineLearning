# Homework 07: Expectation-Maximization Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial as spa
from scipy.stats import multivariate_normal

## Importing Data Set
X = np.genfromtxt("hw07_data_set.csv", delimiter=',')
centroids = np.genfromtxt("hw07_initial_centroids.csv", delimiter=',')

N = len(X)
K = 5

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], s=40, color="k")
plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$x_2$", fontsize=12)

plt.show()


## Algorithm Steps
### Initialization
def initial_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis = 0)
    return(memberships)

def initial_covariance_matrices(memberships, centroids, X, K):
    covariance_matrices = [np.sum([(X[memberships == k][i] - centroids[k]).reshape(2, 1) @ (X[memberships == k][i] - centroids[k]).reshape(1, 2) for i in range(len(X[memberships == k]))], axis=0) / len(X[memberships == k]) for k in range(5)]
    return(covariance_matrices)

def initial_prior_probabilities(memberships, X, N):
    prior_probabilities = np.array(pd.Series(memberships).value_counts().sort_index()/N)
    return(prior_probabilities)

memberships = initial_memberships(centroids, X)
covariance_matrices = initial_covariance_matrices(memberships, centroids, X, K)
prior_probabilities = initial_prior_probabilities(memberships, X, N)

print(f"Initial Centroids (Mean Vectors):\n{centroids}\n")
print(f"Initial Memberships:\n{memberships}\n")
print(f"Membership Value Counts:\n{pd.Series(memberships).value_counts()}\n")
print(f"Initial Covariance Matrices:\n{covariance_matrices}\n")
print(f"Initial Prior Probabilities:\n{prior_probabilities}")

colors = ["C0", "C2", "C3", "C1", "C4"]

plt.figure(figsize=(8, 8))
plt.title("Initial memberships & centroids", fontsize=15)
for i in range(5):
    plt.scatter(X[memberships == i][:, 0], X[memberships == i][:, 1], color=colors[i])
    plt.scatter(centroids[i][0], centroids[i][1], color="k", marker="s", s=50)
plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$x_2$", fontsize=12)
plt.show()

"""
## Expectation-Maximization Algorithm
**loglikelihood:** $L(\Phi|X) = \log\left[\prod_{i=1}^N p(x_i|\Phi)\right]$
$\mathcal{L}(\Phi|X) = \sum_{i=1}^N\left[\sum_{k=1}^N p(x_i|C_k)P(C_k)\right]$

Two sets of random variables:
- $Z = $ cluster memberships (hidden variables)
- $\Phi =$ parameters $[\hat{\mu_1}, \hat{\mu_2}, ..., \hat{\mu_k}, \hat{\Sigma_1}, \hat{\Sigma_2}, ..., \hat{\Sigma_k}]$

**E-STEP:** $E\left[L_c(\Phi|X, Z) | X, \Phi^{(t)}\right]$
**M-STEP:** $\Phi^{t+1} = argmax E\left[L_c(\Phi|X, Z)|X, \Phi^{(t)}\right]$

### Update Equations:
**E-STEP:**
$h_{ik} = E\left[z_{ik}|X, \Phi^{(t)}\right] = \frac{p(x_i|C_k, \Phi^{(t)})P(C_k)}{\sum_{c=1}^K p(x_i|C_c, \Phi^{(t)})P(C_c)}$

**M-STEP:**
$\hat{\mu}_k^{(t+1)} = \frac{\sum_{i=1}^N h_{ik}x_i}{\sum_{i=1}^N h_{ik}}$
$\hat{\sigma}_k^{(t+1)} = \frac{\sum_{i=1}^N h_{ik}(x_i - \hat{\mu}_k^{(t+1)})(x_i - \hat{\mu}_k^{(t+1)})^T}{\sum_{i=1}^N h_{ik}}$
$\hat{P}(C_K) = \frac{\sum_{i=1}^N h_{ik}}{N}$
"""

def multivariate_gaussian(x, mean, covariance):
    return (1. / (np.sqrt((2 * np.pi)**2 * np.linalg.det(covariance))) *  np.exp(-(np.linalg.solve(covariance, x - mean).T.dot(x - mean)) / 2))
def mixture_density(x, mean, covariance, prior_probability):
    return multivariate_gaussian(x, mean, covariance) * prior_probability
def h_ik(x, centroids, covariance_matrices, prior_probabilities):
    return [mixture_density(x, centroids[k], covariance_matrices[k], prior_probabilities[k]) / np.sum([mixture_density(x, centroids[i], covariance_matrices[i], prior_probabilities[i]) for i in range(K)]) for k in range(K)]

### E-Step
def find_memberships(X, centroids, covariance_matrices, prior_probabilities):
    memberships = []

    for m in range(N):
        posterior_probabilities = h_ik(X[m], centroids, covariance_matrices, prior_probabilities)
        max_value = max(posterior_probabilities)
        max_index = posterior_probabilities.index(max_value)
        memberships.append(max_index)
    return np.array((memberships))

### M-Step
def update_centroids(memberships, X):
    return np.vstack([np.sum([h_ik(X[memberships == k][i], centroids, covariance_matrices, prior_probabilities)[k] * X[memberships == k][i] for i in range(len(X[memberships == k]))], axis=0) / len(X[memberships == k]) for k in range(K)])
def update_covariance_matrices(memberships, centroids, X, K):
    return [np.sum([h_ik(X[memberships == k][i], centroids, covariance_matrices, prior_probabilities)[k] * (X[memberships == k][i] - centroids[k]).reshape(2, 1) @ (X[memberships == k][i] - centroids[k]).reshape(1, 2) for i in range(len(X[memberships == k]))], axis=0) / len(X[memberships == k]) for k in range(K)]
def update_prior_probabilities(memberships, X, N):
    return np.array(pd.Series(memberships).value_counts().sort_index()/N)

## Running EM Algorithm
for iteration in range(100):
    memberships = find_memberships(X, centroids, covariance_matrices, prior_probabilities)
    centroids = update_centroids(memberships, X)
    covariance_matrices = update_covariance_matrices(memberships, centroids, X, K)
    prior_probabilities = update_prior_probabilities(memberships, X, N)

    
    print(f"Iteration#:{iteration+1}")

    #for i in range(5):
    #    plt.scatter(X[memberships == i][:, 0], X[memberships == i][:, 1], color=colors[i])
    #    plt.scatter(centroids[i][0], centroids[i][1], color="k", marker="s", s=50)
    #plt.show()

print(f"Mean vectors after 100 iterations:\n\n{centroids}")

## Visualization
initial_means = np.array([[+2.5, +2.5],
                          [-2.5, +2.5],
                          [-2.5, -2.5],
                          [+2.5, -2.5],
                          [+0.0, -0.0]])

initial_covariances = np.array([[[+0.8, -0.6], [-0.6, +0.8]],
                                [[+0.8, +0.6], [+0.6, +0.8]],
                                [[+0.8, -0.6], [-0.6, +0.8]],
                                [[+0.8, +0.6], [+0.6, +0.8]],
                                [[+1.6, +0.0], [+0.0, +1.6]]])

x, y = np.mgrid[-6:+6:.05, -6:+6:.05]
pos = np.dstack((x, y))

original_gausses = [multivariate_normal(mean=initial_means[i], cov=initial_covariances[i]*2) for i in range(5)]
found_gausses = [multivariate_normal(mean=centroids[i], cov=covariance_matrices[i]*2) for i in range(5)]

plt.figure(figsize=(8, 8))
plt.title("Clustering Results & Gaussian Densities", fontsize=15)
for original_gauss in original_gausses:
    plt.contour(x, y, original_gauss.pdf(pos), levels=1, linestyles="dashed", colors="k")
for i, found_gauss in enumerate(found_gausses):
    plt.contour(x, y, found_gauss.pdf(pos), levels=1, colors=colors[i])
for i in range(5):
    plt.scatter(X[memberships == i][:, 0], X[memberships == i][:, 1], color=colors[i])
plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$x_2$", fontsize=12)
plt.show()