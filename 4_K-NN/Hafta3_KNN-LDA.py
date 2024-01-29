# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 00:15:03 2023

@author: samim
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# Iris veri kümesini yükle
data = load_iris()
X = data.data

# Veriyi standartlaştır
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# LDA analizi
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_std, data.target)

# Veriyi eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X_lda, data.target, test_size=0.2, random_state=42)

# KNN modelini oluşturun ve eğitin
knn = KNeighborsClassifier(n_neighbors=3)  # n_neighbors değerini isteğinize göre ayarlayabilirsiniz
knn.fit(X_train, y_train)

# Test seti üzerinde tahmin yapın
y_pred = knn.predict(X_test)

# Doğruluk (accuracy) hesaplayın
accuracy = accuracy_score(y_test, y_pred)
print("KNN Accuracy with LDA:", accuracy)


""" OR """



# # Load the iris dataset
# iris = load_iris()
# data = iris.data

# # Define the target variable
# species = iris.target

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(data, species, test_size=0.33, random_state=0)

# # Create an LDA object
# lda = LinearDiscriminantAnalysis()

# # Fit LDA to the training data
# lda.fit(x_train, y_train)

# # Transform the data using LDA
# x_train_lda = lda.transform(x_train)
# x_test_lda = lda.transform(x_test)

# # Create a K-Nearest Neighbors classifier
# knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# # Fit the classifier to the LDA-transformed training data
# knn.fit(x_train_lda, y_train)

# # Make predictions on the LDA-transformed test data
# result = knn.predict(x_test_lda)

# # Calculate the confusion matrix
# cm = confusion_matrix(y_test, result)
# print(cm)

# # Calculate the accuracy score
# accuracy = accuracy_score(y_test, result)
# print(accuracy)
