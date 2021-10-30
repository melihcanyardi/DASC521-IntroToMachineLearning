# Homework 02: Naïve Bayes Classifier
import numpy as np
import pandas as pd

## Importing Data
images = pd.read_csv('hw02_images.csv', header=None)
labels = pd.read_csv('hw02_labels.csv', header=None)
labels = labels.rename({0: 'Label'}, axis=1)


## Combining Data
data = pd.concat([images, labels], axis=1)
print(f"First five rows of dataset:\n{data.head()}\n")
print(f"Dataset shape: {data.shape}\n")
print(f"Number of unique labels: {data['Label'].nunique()}\n")
print(f"Unique label values: {sorted(data['Label'].unique())}\n")


## Train-Test Split
train = data.iloc[:30000,:]
test = data.iloc[30000:,:]

X_train = train.drop('Label', axis=1)
X_test = test.drop('Label', axis=1)

y_train = train['Label']
y_test = test['Label']

print(f"Train set shape: {train.shape}")
print(f"Test set shape: {test.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}\n")

## Estimating the Parameters
X_1 = train[train['Label'] == 1].drop('Label', axis=1)
X_2 = train[train['Label'] == 2].drop('Label', axis=1)
X_3 = train[train['Label'] == 3].drop('Label', axis=1)
X_4 = train[train['Label'] == 4].drop('Label', axis=1)
X_5 = train[train['Label'] == 5].drop('Label', axis=1)

Xs = [X_1, X_2, X_3, X_4, X_5]

### Sample Mean
#### **Sample mean**: $m =\frac{\sum_{t}x^t}{N}$
def estimate_sample_mean(X):
    return [np.sum(X.iloc[:, i]) / X.shape[0] for i in range(X.shape[1])]

sample_means = np.array([estimate_sample_mean(X) for X in Xs])
print("Sample Means:")
print(sample_means, end='\n\n')

### Standard Deviation
#### **Variance**: $\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; s^2 = \frac{\sum_t{(x^t-m)^2}}{N}$
#### **Standard Deviation**: $\;\;\; s = \sqrt{\frac{\sum_t{(x^t-m)^2}}{N}} $
def estimate_standard_deviation(X, sample_mean):
    return [np.sqrt(np.sum((X.iloc[:, i] - sample_mean[i])**2)/X.shape[0]) for i in range(X.shape[1])]

sample_deviations = np.array([estimate_standard_deviation(X, sample_mean) for X, sample_mean in zip(Xs, sample_means)])
print("Sample Deviations:")
print(sample_deviations, end='\n\n')

### Prior Probabilities
#### **Prior Probability**: $\hat{P}(C_i) = \frac{\sum_t{r_i^t}}{N}$
def prior_probability(X, all_X):
    return X.shape[0] / all_X.shape[0]

class_priors = [prior_probability(X, train) for X in Xs]
print("Class Priors:")
print(class_priors, end='\n\n')


## Naive Bayes
### Alpaydin, E., Introduction to Machine Learning, Section 4.2.3: Gaussian (Normal) Density
#### **Log likelihood of Gaussian (Normal) Density:** $L(\mu, \sigma | X) = -\frac{N}{2}\log{(2\pi)} - N\log{\sigma} - \frac{\sum_t{(x^t-\mu)^2}}{2\sigma^2}$

# $\newline\newline$
 
### Alpaydin, E., Introduction to Machine Learning, Section 4.5: Parametric Classification 
#### **Discriminant Function:** $g_i(x) = p(x|C_i)P(C_i)$
#                               or
####                            $g_i(x) = \log{p(x|C_i)} + \log{P(C_i)}$
#                               where
####                            $p(\textbf{x}|C_i) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(x-\mu_i)^2}{2\sigma_i^2}\right]}$
# $\newline\newline$
# 
#### **Parametric Classification:** $g_i(\textbf{x}) = -\frac{1}{2}\log{2\pi} - \log{\sigma_i}-\frac{(x-\mu_i)^2}{2\sigma_i^2} + \log{P(C_i)}$ 
##### Plugging the estimates for the **means** $\left(m =\frac{\sum_{t}x^t}{N}\right)$, **variance** $\left(s^2 = \frac{\sum_t{(x^t-m)^2}}{N}\right)$,
##### and **priors** $\left(\hat{P}(C_i) = \frac{\sum_t{r_i^t}}{N}\right)$, we get:
#### $g_i(\textbf{x}) = -\frac{1}{2}\log{2\pi} - \log{s_i}-\frac{(x-m_i)^2}{2s_i^2} + \log{\hat{P}(C_i)}$

def discriminant_function(x, sample_mean, sample_deviation, class_prior):
    return np.sum((-1/2 * np.log(2*np.pi)) - (np.log(sample_deviation)) - ((x - sample_mean)**2) / (2 * sample_deviation**2) + np.log(class_prior))


## Classification Algorithm
##### Calculate scores of x's in the training set for each class. Pick the class with the highest score.

def predict(x):
    
    scores = []
    
    for i in range(5):
        scores.append(discriminant_function(x, sample_means[i], sample_deviations[i], class_priors[i]))
        
    scores = pd.Series(scores)
    return scores[scores == np.max(scores)].index[0] + 1

### Predictions
y_pred_train = np.array([predict(X_train.iloc[i, :]) for i in range(X_train.shape[0])])
y_pred_test = np.array([predict(X_test.iloc[i, :]) for i in range(X_test.shape[0])])

## Calculating Confusion Matrix
### Train Set
confusion_matrix_train = pd.crosstab(y_pred_train, y_train, rownames = ["y_pred"], colnames = ["y_truth"])

print("Confusion Matrix - Training Set:")
print(confusion_matrix_train)

### Test Set
confusion_matrix_test = pd.crosstab(y_pred_test, y_test, rownames = ["y_pred"], colnames = ["y_truth"])

print("Confusion Matrix - Test Set:")
print(confusion_matrix_test)