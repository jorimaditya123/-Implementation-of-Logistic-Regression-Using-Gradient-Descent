# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the placement dataset.
2. Scale the features and split the data into training and testing sets.
3. Initialize weights and bias, then update them using Gradient Descent with the sigmoid function.
4. Predict the test results and evaluate the model using accuracy, confusion matrix, and classification report.
 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Aditya Jorim F S
RegisterNumber:  212225240004
*/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('/content/drive/MyDrive/Placement_Data.csv')

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])


# Features and Target
X = df.drop(['sl_no', 'status', 'salary'], axis=1)
y = df['status']

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

weights = np.zeros(X_train.shape[1])
bias = 0

learning_rate = 0.01
epochs = 1000

for i in range(epochs):

    linear_model = np.dot(X_train, weights) + bias
    predictions = sigmoid(linear_model)

    dw = (1 / len(X_train)) * np.dot(X_train.T, (predictions - y_train))
    db = (1 / len(X_train)) * np.sum(predictions - y_train)

    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

linear_test = np.dot(X_test, weights) + bias
predictions = sigmoid(linear_test)
y_pred = [1 if i >= 0.5 else 0 for i in predictions]


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Output:
<img width="1081" height="420" alt="image" src="https://github.com/user-attachments/assets/0b46f1b6-f81b-40be-8823-1ab1ec7bde5f" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

