import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Data Set
data_set = pd.read_csv('hw04_data_set.csv')

# Train-Test Split
train = data_set[:150]
test = data_set[150:]

X_train = train['eruptions']
y_train = train['waiting']
X_test = test['eruptions']
y_test = test['waiting']

N = data_set.shape[0]


# Regressogram
bin_width = 0.37
origin = 1.5

minimum_value = origin
maximum_value = 5.2

left_borders = np.arange(minimum_value, maximum_value, bin_width)
print(f"Left Borders:\n{left_borders}")
print(f"Length of Left Borders: {len(left_borders)}\n")

right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)
print(f"Right Borders:\n{right_borders}")
print(f"Length of Right Borders: {len(right_borders)}\n")

g_hat = [np.sum((left_borders[i] < X_train) & (X_train <= right_borders[i])) * y_train[(left_borders[i] < X_train) & (X_train <= right_borders[i])].mean() / np.sum((left_borders[i] < X_train) & (X_train <= right_borders[i])) for i in range(len(left_borders))]
print(f"g_hat:\n{g_hat}")

## Drawing Regressogram
plt.figure(figsize=(10, 6))

plt.scatter(train['eruptions'], train['waiting'], c='b', label='training')
plt.scatter(test['eruptions'], test['waiting'], c='r', label='test')
plt.xlabel("Eruption time (mm)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc='upper left')

for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [g_hat[b], g_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [g_hat[b], g_hat[b + 1]], "k-")    

plt.show()

# RMSE of Regressogram
def calculate_rmse(X_test, y_test, left_borders, right_borders):
    total_error = sum([((y_test[(left_borders[i] < X_test) & (X_test <= right_borders[i])] - g_hat[i])**2).sum() for i in range(len(left_borders))])
    return np.sqrt(total_error / len(X_test))

rmse = calculate_rmse(X_test, y_test, left_borders, right_borders)

print(f"Regressogram => RMSE is {rmse} when h is {bin_width}")


# Running Mean Smoother
bin_width = 0.37
data_interval = np.linspace(minimum_value, maximum_value, 1601)

def w(u):
    if np.abs(u) < 1:
        return 1
    else:
        return 0

g_hat = [(np.asarray([w(u) for u in (x - X_train) / bin_width]) * y_train).sum() / (np.asarray([w(u) for u in (x - X_train) / bin_width])).sum() for x in data_interval]

## Drawing Running Mean Smoother
plt.figure(figsize=(10, 6))

plt.scatter(train['eruptions'], train['waiting'], c='b', label='training')
plt.scatter(test['eruptions'], test['waiting'], c='r', label='test')
plt.xlabel("Eruption time (mm)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc='upper left')

plt.plot(data_interval, g_hat, "k-")

plt.show()

# RMSE of Running Mean Smoother
left_borders = data_interval[:-1]
right_borders = data_interval[1:]

rmse = calculate_rmse(X_test, y_test, left_borders, right_borders)

print(f"Running Mean Smoother => RMSE is {rmse} when h is {bin_width}")


# Kernel Smoother
bin_width = 0.37
data_interval = np.linspace(minimum_value, maximum_value, 1601)

def K(u):
    return 1/np.sqrt(2*np.pi) * np.exp(-(u**2)/2)

g_hat = [(np.asarray([K(u) for u in (x - X_train) / bin_width]) * y_train).sum() / (np.asarray([K(u) for u in (x - X_train) / bin_width])).sum() for x in data_interval]

## Drawing Kernel Smoother
plt.figure(figsize=(10, 6))

plt.scatter(train['eruptions'], train['waiting'], c='b', label='training')
plt.scatter(test['eruptions'], test['waiting'], c='r', label='test')
plt.xlabel("Eruption time (mm)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc='upper left')

plt.plot(data_interval, g_hat, "k-")

plt.show()

# RMSE of Kernel Smoother
left_borders = data_interval[:-1]
right_borders = data_interval[1:]

rmse = calculate_rmse(X_test, y_test, left_borders, right_borders)

print(f"Kernel Smoother => RMSE is {rmse} when h is {bin_width}")