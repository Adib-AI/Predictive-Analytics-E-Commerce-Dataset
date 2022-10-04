#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVR
from sklearn.metrics import classification_report, mean_squared_error


# In[2]:


df = pd.read_csv('G:/Kumpulan Dataset/E_commerce_regression/Ecommerce Customers', sep = ',')


# In[3]:


df


# # EDA

# In[4]:


df.info()


# Tidak terdapat data NaN dan Null

# In[5]:


df.describe().T


# In[6]:


df.select_dtypes(include = 'number')


# In[7]:


fig, axes = plt.subplots(1,5, figsize = (15,5))

for i, ax in zip(df.select_dtypes(include = 'number'), axes.flatten()):
    sns.boxplot(x = df[i], ax = ax)
    plt.tight_layout()


# Handle Outlier

# In[8]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis = 1)]

df.shape


# Disini saya tetap menggunakan outlier, meskipun data yang dimiliki sangat kecil. Saya tidak mengganti data pada nilai outliernya.
# Pada label yang digunakan, saya akan memprediksi nilai pada kolom Yearly Amount Spent

# # Univariate Analysis 

# In[9]:


numerical_kolom = df.select_dtypes(include = 'number').columns.tolist()
categorical_kolom = df.select_dtypes(include = 'object').columns.tolist()

print("Numerical kolom : {}".format(numerical_kolom))
print("Categorical kolom : {}".format(categorical_kolom))


# In[10]:


df.nunique()


# Ternyata semua kolom (kecuali Avatar) memiliki nilai unique per row nya. Sehingga, saya akan langsung melakukan pengecekan ke data yang besifat numerical

# ## Numerical Kolom 

# In[11]:


df.hist(bins = 50, figsize=(20,15))
plt.show()


# Kesimpulan yang didapat:
# 1. Waktu penggunaan di website lebih lama daripada penggunaan di aplikasi dengan rata-rata penggunaan sekitar 32-34 menit.
# 2. Namun, pada penggunaan aplikasi memberikan kenaikan yang positif pada amount spent
# 2. Skala lamanya membership di range 3 dan 4 bulan memiliki amount spent yang tinggi.
#     
# Sehingga, transaksi sering dilakukan pada penggunaan sistem aplikasi dengan waktu yang relatif singkat.

# # Multivariate Analysis

# ## Numerical Kolom

# In[12]:


sns.pairplot(df, diag_kind = 'kde')


# In[13]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot = True, annot_kws = {'fontsize':20}, cmap = 'spring_r', linewidth = 0.3)
plt.title('Correlation each feature', fontsize = 20)
plt.show()


# # Preprocessing

# ## Reduction Kolom 

# Disini saya tidak menggunakan PCA dikarenakan tidak adanya korelasi yang tinggi antar fitur yang sama. 
# Menurut perkiraan saya, Time On Website dengan Avg Session dapat dilakukan PCA. Namun dengan korelasi yang cukup rendah. 
# Hal tersebut tidak perlu dilakukan dan yang saya gunakan hanyalah korelasi dengan rentang mendekati -1 dan +1
# 
# Untuk pembagian dataset, saya menggunakan 75% (Train) : 25% (Test) karena mengingat dataset yang kecil

# In[14]:


X = df[['Avg. Session Length', 'Time on App', 'Length of Membership']]
y = df['Yearly Amount Spent']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 250)


# In[16]:


print(f'Total of Dataset : {len(X)}')
print(f'Total of Train Dataset : {len(X_train)}')
print(f'Total of Test Dataset : {len(X_test)}')


# ## Standardization 

# In[17]:


kolom = ['Avg. Session Length', 'Time on App', 'Length of Membership']


# In[18]:


scaler = StandardScaler()
scaler.fit(X_train[kolom])
X_train[kolom] = scaler.transform(X_train.loc[:, kolom])
X_train[kolom].head()


# In[19]:


X_test.loc[:,kolom] = scaler.transform(X_test.loc[:,kolom])


# In[20]:


X_test


# # Modeling 

# In[21]:


model = SVR(kernel = 'linear')
model.fit(X_train, y_train)


# In[22]:


print('Mean Squared Error pada data train : {}'.format(mean_squared_error(y_pred = model.predict(X_train), y_true = y_train)))


# In[23]:


print('Mean Squared Error pada data train : {}'.format(mean_squared_error(y_pred = model.predict(X_test), y_true = y_test)))


# Disini saya akan mencoba parameter optimasi parameter kernel linear yaitu C

# In[24]:


for c in range(1, 21):
    models = SVR(kernel = 'linear', C = c)
    models.fit(X_train, y_train)
    
    print('Mean Squared Error pada data train : {} (C = {})'.format(mean_squared_error(y_pred = models.predict(X_train), y_true = y_train).round(2), c))


# Hasil MSE terkecil dihasilkan ketika nilai C di rentang 14-20 dengan nilai 103.07.
# Saya akan menghitung nilai MSE dengan X_test

# In[25]:


for c in range(1, 21):
    models = SVR(kernel = 'linear', C = c)
    models.fit(X_train, y_train)
    
    print('Mean Squared Error pada data train : {} (C = {})'.format(mean_squared_error(y_pred = models.predict(X_test), y_true = y_test).round(2), c))


# In[26]:


x_prediksi_nilai_rill = X_test.iloc[0:3].copy()
y_prediksi_nilai_rill = y_test.iloc[0:3].copy()


# In[27]:


x_prediksi_nilai_rill


# In[28]:


y_prediksi_nilai_rill


# Model tanpa optimizer nilai C

# In[29]:


model.predict(x_prediksi_nilai_rill)


# Model dengan optimizer nilai C dengan mse terkecil, yaitu C = 7-9

# In[30]:


models.predict(x_prediksi_nilai_rill)


# In[31]:


data = y_prediksi_nilai_rill.values, model.predict(x_prediksi_nilai_rill), models.predict(x_prediksi_nilai_rill)


# In[32]:


df_evaluation = pd.DataFrame(data = data, index = ['Nilai Rill', 'Tanpa C', 'Dengan C'])


# In[33]:


df_evaluation.T


# Terdapat beberapa kernel pada Support Vector antara lain, yaitu linear, RBF, polynominal, sigmoid.
# Pada kasus diatas, kernel linear sudah dapat memberikan hasil yang mendekati dengan hasil sebenarnya dengan bantuan parameternya. Hal ini saya perkirakan karena jumlah N-fitur yang lebih rendah daripada jumlah N-rows pada data.
