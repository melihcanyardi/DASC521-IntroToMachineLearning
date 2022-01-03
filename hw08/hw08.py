# Homework 08: Spectral Clustering

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa

## Importing Data Set
X = np.genfromtxt("hw08_data_set.csv", delimiter=',')

N = X.shape[0]

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], s=40, color="k")
plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$x_2$", fontsize=12)
plt.show()

## Euclidean Distance & Connectivity Matrix
def euclidean_distance(x1, x2):
    return np.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2)

euc = np.array([[euclidean_distance(X[i], X[k]) for k in range(N)] for i in range(N)])

threshold = 1.25
B = np.array([[1 if euc[i, k] < threshold and i != k else 0 for k in range(N)] for i in range(N)])

print(f"Euclidean Distances:\n{euc}\n")
print(f"B matrix:\n{B}\n")

### Visualization of Connectivity Matrix
plt.figure(figsize=(8, 8))
for i in range(N):
    for k in range(N):
        if B[i, k] == 1 and i != k:
            plt.plot(np.array([X[i, 0], X[k, 0]]), np.array([X[i, 1], X[k, 1]]), linewidth=0.5, c="grey")
plt.scatter(X[:, 0], X[:, 1], s=50, color="k", zorder=3)
plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$x_2$", fontsize=12)
plt.show()

## Degree, Laplacian & Normalized Laplacian Matrices
D = np.zeros((300, 300), dtype=int)

for i in range(300):
    D[i, i] = B[i].sum()
    
print(f"D Matrix:\n{D}\n")

L = D - B
print(f"L Matrix:\n{L}\n")

D_new = np.zeros((300, 300))
for i in range(300):
    D_new[i, i] = D[i, i]**(-1/2)

L_symmetric = np.eye(300) - (D_new @ B @ D_new)
print(f"Normalized Laplacian Matrix:\n{L_symmetric}\n")

## Eigenvectors of Normalized Laplacian Matrix
R = 5
eig_vals, eig_vecs = np.linalg.eig(L_symmetric)
Z = eig_vecs[:,np.argsort(eig_vals)[1:R+1]]
print(f"Z Matrix:\n{Z}\n")

centroids = np.vstack([Z[28], Z[142], Z[203], Z[270], Z[276]])
print(f"Initial Centroids:\n{centroids}\n")

## K-Means Algorithm
def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis = 0)
    return(memberships)

def update_centroids(memberships, X):
    centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)

memberships = update_memberships(centroids, Z)
iteration = 1
K = 5

while True:
    print(f"Iteration #{iteration}")
    print(f"Centroids:\n{centroids}")
    print(f"Memberships:\n{memberships}\n")
    old_centroids = centroids
    centroids = update_centroids(memberships, Z)
    if np.alltrue(centroids == old_centroids):
        break
    old_memberships = memberships
    memberships = update_memberships(centroids, Z)
    iteration += 1
    
centroids = update_centroids(memberships, X)

## Visualization of Clustering results
colors = ["C0", "C2", "C3", "C1", "C4"]
plt.figure(figsize=(8, 8))
for c in range(K):
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 15, color=colors[c])
for c in range(K):
    plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12, 
             markerfacecolor = colors[c], markeredgecolor = "black")
plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$x_2$", fontsize=12)
plt.show()