""" Temel Temizleme """

import pandas as pd
import numpy as np

df = pd.DataFrame({
'SözleşmeID': [101, 102, 103, 104, 105, 106, 107, 108],
'MüşteriSüre': [10, 11, 12, 10, 11, 10, 12, 10],
'Şehir': ['Bursa', 'Bursa', 'Kocaeli', 'Bursa', 'Bursa', 'Bursa', 'Bursa', 'Bursa'],
'FaturaTutar': ['100', '120', '110', '90', '80', '150', '45', '65'],
'Ülke': ['Türkiye','Türkiye','Türkiye','Türkiye','Türkiye','Türkiye','Türkiye','Türkiye'],
'id_group':[7,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN]
})


def oznitelikSecme():
  # 1) Kayıp Değerleri olan değişkeni çıkart
  df2 = df.loc[:, df.isnull().mean() < .8]
  # 2) Tüm değerleri aynı olan kategorik değişkeni çıkart
  df2 = df2.loc[:, df2.nunique() != 1 ]
  # 3) Tüm değerleri farklı olan kategorik değişkeni çıkart
  df2 = df2.loc[:, df2.nunique() != len(df2)]
  return df2



print(df.dtypes)
print('----------------')
print(df.nunique())
print('----------------')
print(df.shape)
print('----------------')
df0 = oznitelikSecme()
print(df0)
