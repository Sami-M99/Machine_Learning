
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

# هنا استدعينا قاعدة البيانات من مكتبة sklearn.datasets 
#  iris veri seti 3 farklı sınıfa ait 4 öznitelik  bulunmaktadır.
iris=load_iris()
# هنا نحولها الى بيانات data 
data=iris.data
print(type(data))     # ndarray ==> N boyutlu bir dizi türüdür
# هنا اخذنا عناوين الأربع الأعمدة للبيانات öznitelik 
feature_names=iris.feature_names
print(feature_names)
# وهنا اخذنا المخرجات المكونة من ثلاثة أنواع 0,1,2 sınıf 
y=iris.target
print("-------------------------")

# هنا جلنا البيانات ك DataFrame فاخنا البيانات من data واسماء أعمدة البيانات من feature_names 
df=pd.DataFrame(data,columns=feature_names)
# اضفنا عمود "خاصية" جديد للجدول بالمخرجات 
df["sınıf"]=y
# وهنا عملنا متغير للمدخلات 
x=data
print(df.head())


# من هنا نبدأ برسم الأشكال البيانية 
fig, axs = plt.subplots(2, 2)
row,col=0,0

#1-4 feature deneniyor. Amaç boyut indirgeme olduğunda n_component=2 en uygunu
for i in range(1,5):
    pca=PCA(n_components=(i),whiten=(False))  #whiten->normalize
    x_pca=pca.fit(x)
    print("\nn_components=",i)
    print("Variance ratio=",pca.explained_variance_ratio_)  # Fark oran
    print("Sum ratio=",sum(pca.explained_variance_ratio_))  # Toplam oran

        
    if (i>1):      
        axs[row,col].set(xlabel='Number of Component', ylabel='Cumulative Variance')
        axs[row,col].label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.
        axs[row,col].plot(np.cumsum(pca.explained_variance_ratio_)) #cumsum kümülatif toplamı bulur. Her bileşen için toplama ekler
        col+=1
        if (col==2):
            row+=1
            col=0
    
    if (i==4):
        df_sns=pd.DataFrame({'variance':pca.explained_variance_ratio_,'PC':['PC1','PC2','PC3','PC4']})
        sns.barplot(x='PC',y='variance',data=df_sns,color="c")
        plt.show()
    


