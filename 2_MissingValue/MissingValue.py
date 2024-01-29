import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import copy


data=pd.read_csv('bikedetails.csv')
# عمل نسخة عميقة للبيانات
orginaldata=copy.deepcopy(data)
# يطبع لنا نوع الجدول DataFrame وعدد الصفوف وعدد الأعمدة واسمائها واستخدام الذاكرة 
print(data.info())
print("-------------------------")
# طباعة اسماء الأعمدة مع عدد البيانات الفارغة لكل عمود
print(data.isnull().sum())
print(data.shape)
print("-------------------------")
print("------------ 1# Delete Rows with Missing Values (Eksik Değerler) -------------")
print("-------------------------")
# هذه الطرية الأولى للتخلص من البيانات الفارغة بحذها 
# هنا نقوم بحذف البيانات الفارغة بحذف السطر اذا فيه خانة واحدة فارغة او اكثر
# وبذلك بدل ما كانت البيانات 1061 سطر صارت 626 لوجود 435 بيان فارغ
# inplace == yeni data mı oluşturulacak  
data.dropna(inplace=True)
print(data.info())
print(data.isnull().sum())
print(data.shape)


print("-------------------------")
print("------------ 2# Imputation (Eksik Verileri Tamamlama)  -------------")
print("-------------------------")
# هذه الطريقة الثانية لتعديل البانات الفارغة بتعبائتها بقيم متوسطة من البيانات الموجودة 
#--------mean imputer-----------------------------
#---------Method-1
meandata=pd.read_csv('bikedetails.csv')
# هنا اخترنا العمود الذي فيه البيانات الفارغة 
print(meandata["ex_showroom_price"][:20])
meandata["ex_showroom_price"]=meandata["ex_showroom_price"].replace(np.NAN,meandata["ex_showroom_price"].mean())
print(meandata["ex_showroom_price"][:20])
print("-------------------------")

#--------Method-2
meandata=pd.read_csv('bikedetails.csv')
print(meandata["ex_showroom_price"][:20])
# SimpleImputer adlı bir sınıfı kullanarak eksik verileri doldurur.
fea_transformer = SimpleImputer(strategy="mean")
#  يقوم بتنفيذ استراتيجية ملء القيم المفقودة عن طريق استخدام عمود "ex_showroom_price"يتم تخزينها في values.
values = fea_transformer.fit_transform(meandata[["ex_showroom_price"]])
# هنا حدثنا قيم العمود للقيم التي بعمود value 
meandata["ex_showroom_price"]=pd.DataFrame(values)
print(meandata["ex_showroom_price"][:20])
print("-------------------------")
print("-------------------------")
#------------------------------------

#--------median imputer-----------------------------
#---------Method-1
mediandata=pd.read_csv('bikedetails.csv')
print(mediandata["ex_showroom_price"][:20])
mediandata["ex_showroom_price"]=mediandata["ex_showroom_price"].replace(np.NAN,mediandata["ex_showroom_price"].median())
print(mediandata["ex_showroom_price"][:20])
print("-------------------------")
#--------Method-2
mediandata=pd.read_csv('bikedetails.csv')
print(mediandata["ex_showroom_price"][:20])
fea_transformer = SimpleImputer(strategy="median")
values = fea_transformer.fit_transform(mediandata[["ex_showroom_price"]])
mediandata["ex_showroom_price"]=pd.DataFrame(values)
print(mediandata["ex_showroom_price"][:20])
print("-------------------------")
print("-------------------------")
#------------------------------------


#----------KNN Imputer------------
from sklearn.impute import KNNImputer
 
knndata=pd.read_csv('bikedetails.csv')
print(knndata["ex_showroom_price"][:20])
# أحدد أقرب جار ليكون 3 لازم يكون رقم فردي 
fea_transformer = KNNImputer(n_neighbors=3)
values = fea_transformer.fit_transform(knndata[["ex_showroom_price"]])
knndata["ex_showroom_price"]=pd.DataFrame(values)
print(knndata["ex_showroom_price"][:20])


