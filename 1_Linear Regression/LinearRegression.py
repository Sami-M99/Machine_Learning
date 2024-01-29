from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


#  قراءة البيانات 
veriseti = pd.read_csv('linear_model.csv')
# اول ما يتم قراءة البيانات من csv تكون من نوع DataFrame 
print(type(veriseti))    # ==> <class 'pandas.core.frame.DataFrame'>
# طباعة اول 5 اسطر
print(veriseti.head()) 

print("\n---null veri seti sayısı----")
print(veriseti.isnull().sum()) 


# ملئ البيانات الفارغة بعمود y بمتوسط القيم
veriseti.y.fillna(value=veriseti.y.mean(), inplace=True)

# اعطاء معلومات حول البيانات المعطاة
print("\n----veri seti describe---")
print(veriseti.describe())

# يتم تطبيق تطبيع MinMax يعني يجعل البيانات بين 0 و 1 
scaler = MinMaxScaler()
scaler.fit(veriseti)
veriseti = pd.DataFrame(scaler.transform(veriseti)) #minmax normalizasyon uygulanıyor

# نعيد تسمية الأعمدة ل x ,y لانة عند التحويل تغيرت اسماء الاعمدة الى 0, 1 
veriseti.columns=["x","y"] 
print(veriseti.describe())

x=veriseti.x
y=veriseti.y
 # الأن نوع  البيانات ل x,y هو series ( 100 ,) يعني عمود بالقيم مع اسمة x او y  
# ولكن هنا اخذنا فقط القيم الموجودة بالعمود وخلينا شكلها مصفوفة احادية Array (100 , 1) 
#iki boyutlu numpy array'e dönüştürülüyor ilk boyuttaki eleman sayısına 100'de yazılabilir. -1 yazılırsa kendi hesaplar
x=x.values.reshape(-1,1) 
y=y.values.reshape(-1,1) 

#x ve y arasındaki ilişkiye plotter olarak bakalım
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('Y')
plt.title('x ve y arasındaki ilişki')
plt.show()

# -------  Linear Regression  -------
lineer_regresyon = LinearRegression()
lineer_regresyon.fit(x,y)

#---model bilgileri yazdırılıyor
print("\nModel Information")
print(lineer_regresyon.intercept_) #parameter m
print(lineer_regresyon.coef_)      #parameter b
print("Elde edilen regresyon modeli: Y={}+{}X".format(lineer_regresyon.intercept_,lineer_regresyon.coef_[0]))


# model test
print("\nTest Model")
y_predicted = lineer_regresyon.predict(x)
#  مقياس R-squared (R2) هو مقياس لمدى التباين في المتغير التابع الذي يتم تفسيره بواسطة المتغيرات المستقلة.
#  كلما كانت القيمة أقرب إلى 1، كلما كان النموذج يشرح البيانات بشكل أفضل.
print("R2=",r2_score(y,y_predicted))
#   مقياس MSE ortalama karesel hata كلما كان قريب لل 0 كان أفضل 
print("MSE=",mean_squared_error(y, y_predicted))


plt.scatter(x, y,color='red') #ham verinin dağılımı kırmızı noktalar.
plt.plot(x, y_predicted, color='blue',label='regresyon grafiği') # modelin tahmin ettiği mavi çizgi
plt.xlabel('x')
plt.ylabel('Y')
plt.title('X y regresyon analiz')
plt.show()
