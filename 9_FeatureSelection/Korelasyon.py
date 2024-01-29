""" Korelasyon """

from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


x = fetch_california_housing()
df = pd.DataFrame(x.data, columns = x.feature_names)
df["Longitude"] = x.target
X = df.drop(columns="Longitude")
#Özellikler
y = df["Longitude"]
#Hedef Değişken
print(df.head())
#Pearson Korelasyon
plt.figure(figsize=(12, 10))
cor = df.corr() #korelasyon hesaplar
print(cor)
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
#Hedef değişken ile olan korelasyon
cor_target = abs(cor["Longitude"]) #korelasonu yönü önemli olmadığı şiddeti bizim için önemli o yüzden mutlak değerini alıyoruz.
#Hedef değişkenle yüksek korelasyonlu olanları seç
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)

print("-------------------")

# Yüksek korelasyonlu özelliklerin isimlerini al
relevant_feature_names = relevant_features.index.tolist()

# Yüksek korelasyonlu özellikleri içeren yeni veri setini oluştur
new_df = df[relevant_feature_names]
print("\nYeni Veri Seti Örnekleri:")
print(new_df.head())

new_df = new_df.drop(columns="Longitude")
print("\nYeni Veri Seti Örnekleri:--------------")
print(new_df.head())
