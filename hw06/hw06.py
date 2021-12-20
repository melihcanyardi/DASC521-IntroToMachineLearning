# Homework 06: One-Versus-All Support Vector Classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance as dt
import cvxopt as cvx
from sklearn.metrics import accuracy_score

## Importing Data Set
images = pd.read_csv("hw06_images.csv", header=None)
labels = pd.read_csv("hw06_labels.csv", header=None)

## Train-Test Split
X_train = images[:1000]
y_train = np.array(labels[:1000][0])
X_test = images[1000:]
y_test = np.array(labels[1000:][0])

N_train = len(y_train)
N_test = len(y_test)

## Distance and Kernel Functions (Gaussian Kernel)
def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)

## Learning Algorithm
### Train Set
def one_vs_all_classification(C, class_num):
    y_train_ = y_train.copy()
    y_train_[y_train_ != class_num] = -1
    y_train_[y_train_ == class_num] = 1
    s = 10
    K_train = gaussian_kernel(X_train, X_train, s)
    yyK = np.matmul(y_train_[:,None], y_train_[None,:]) * K_train
    
    C = C
    epsilon = 1e-3
    print(f"C = {C}, class = {class_num}")
    
    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
    A = cvx.matrix(1.0 * y_train_[None,:])
    b = cvx.matrix(0.0)
    
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_train)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C
    
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y_train_[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
    
    f_predicted = np.matmul(K_train, y_train_[:,None] * alpha[:,None]) + w0

    return f_predicted

def predict(C):
    f_predicted_1 = one_vs_all_classification(C=C, class_num=1)
    f_predicted_2 = one_vs_all_classification(C=C, class_num=2)
    f_predicted_3 = one_vs_all_classification(C=C, class_num=3)
    f_predicted_4 = one_vs_all_classification(C=C, class_num=4)
    f_predicted_5 = one_vs_all_classification(C=C, class_num=5)
    
    y_predicted = []
        
    for i in range(1000):
        predictions = [f_predicted_1[i], f_predicted_2[i], f_predicted_3[i], f_predicted_4[i], f_predicted_5[i]]
        max_value = max(predictions)
        max_index = predictions.index(max_value)
        y_predicted.append(max_index + 1)

    return y_predicted

y_predicted = predict(C=10)

confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)

train_predictions = []
for C in [0.1, 1, 10, 100, 1000]:
    train_predictions.append(predict(C))

train_accuracies = []

for prediction in train_predictions:
    train_accuracies.append(accuracy_score(y_train, prediction)),

### Test Set
def one_vs_all_classification(C, class_num):
    y_test_ = y_test.copy()
    y_test_[y_test_ != class_num] = -1
    y_test_[y_test_ == class_num] = 1
    s = 10
    K_test = gaussian_kernel(X_test, X_test, s)
    yyK = np.matmul(y_test_[:,None], y_test_[None,:]) * K_test
    
    C = C
    epsilon = 1e-3
    print(f"C = {C}, class = {class_num}")
    
    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_test, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_test), np.eye(N_test))))
    h = cvx.matrix(np.vstack((np.zeros((N_test, 1)), C * np.ones((N_test, 1)))))
    A = cvx.matrix(1.0 * y_test_[None,:])
    b = cvx.matrix(0.0)
    
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_test)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C
    
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y_test_[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
    
    f_predicted = np.matmul(K_test, y_test_[:,None] * alpha[:,None]) + w0

    return f_predicted

def predict(C):
    f_predicted_1 = one_vs_all_classification(C=C, class_num=1)
    f_predicted_2 = one_vs_all_classification(C=C, class_num=2)
    f_predicted_3 = one_vs_all_classification(C=C, class_num=3)
    f_predicted_4 = one_vs_all_classification(C=C, class_num=4)
    f_predicted_5 = one_vs_all_classification(C=C, class_num=5)
    
    y_predicted = []
        
    for i in range(4000):
        predictions = [f_predicted_1[i], f_predicted_2[i], f_predicted_3[i], f_predicted_4[i], f_predicted_5[i]]
        max_value = max(predictions)
        max_index = predictions.index(max_value)
        y_predicted.append(max_index + 1)

    return y_predicted

y_predicted = predict(C=10)

confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_test), y_test, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)

test_predictions = []
for C in [0.1, 1, 10, 100, 1000]:
    test_predictions.append(predict(C))

test_accuracies = []

for prediction in test_predictions:
    test_accuracies.append(accuracy_score(y_test, prediction)),

## Visualization
plt.figure(figsize=(12, 8))

plt.plot([0.1, 1, 10, 100, 1000], train_accuracies, color="blue", marker="o", label="training")
plt.plot([0.1, 1, 10, 100, 1000], test_accuracies, color="red", marker="o", label="testing")

plt.ticklabel_format(style='sci', axis='x', scilimits=(10**-1, 10**3))
plt.xticks(ticks=[0.1, 1, 10, 100, 1000], labels=[f"$10^-1$", f"$10^0$", f"$10^1$", f"$10^2$", f"$10^3$"])

plt.xlabel("Regularization parameter (C)")
plt.ylabel("Accuracy")

plt.legend()

plt.xticks()
plt.show()