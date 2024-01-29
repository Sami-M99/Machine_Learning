import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Iris veri kümesini yükle
data = load_iris()
X = data.data

# Veriyi standartlaştır
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# PCA analizi
pca = PCA()
X_pca = pca.fit_transform(X_std)

# Veriyi eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X_pca, data.target, test_size=0.2, random_state=42)

# KNN modelini oluşturun ve eğitin
knn = KNeighborsClassifier(n_neighbors=3)  # n_neighbors değerini isteğinize göre ayarlayabilirsiniz
knn.fit(X_train, y_train)

# Test seti üzerinde tahmin yapın
y_pred = knn.predict(X_test)

# Doğruluk (accuracy) hesaplayın
accuracy = accuracy_score(y_test, y_pred)
print("KNN Accuracy:", accuracy)



"""   OR """

# # csv dosyamızı okuduk.
# iris = load_iris()
# data = iris.data

# # Bağımlı Değişkeni (species) bir değişkene atadık
# species = iris.target

# x_train, x_test, y_train, y_test = train_test_split(data, species, test_size=0.33, random_state=0)

# # PCA uygulama
# pca = PCA(n_components=2)  # İki temel bileşeni kullanacağız
# x_train_pca = pca.fit_transform(x_train)
# x_test_pca = pca.transform(x_test)

# # KNeighborsClassifier sınıfından bir nesne ürettik
# knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
# knn.fit(x_train_pca, y_train)

# # Test veri kümemizi verdik ve iris türü tahmin etmesini sağladık
# result = knn.predict(x_test_pca)

# # Karmaşıklık matrisi
# cm = confusion_matrix(y_test, result)
# print(cm)

# # Başarı Oranı
# accuracy = accuracy_score(y_test, result)
# print(accuracy)
