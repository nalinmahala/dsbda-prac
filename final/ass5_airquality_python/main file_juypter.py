#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


air = pd.read_csv('airquality.csv')


# In[3]:


air.shape


# In[4]:


air.head()


# In[5]:


air.count()


# In[6]:


air.isnull().sum()


# In[7]:


air.describe()


# In[8]:


air.info()


# In[9]:


A = air.dropna()


# In[10]:


A.shape


# In[11]:


A = air.fillna(0)


# In[12]:


A.shape


# In[13]:


A.head()


# In[14]:


A = air.fillna(method='pad')


# In[15]:


A.head()


# In[16]:


A = air.fillna(method='backfill')


# In[17]:


A.head()


# In[18]:


import numpy as np


# In[19]:


A = air['Ozone'].replace(np.NaN,air['Ozone'].mean())


# In[20]:


A.head()


# In[21]:


A = air['Ozone'].replace(np.NaN,air['Ozone'].mean())


# In[22]:


A.head()


# In[23]:


A = air['Ozone'].replace(np.NaN,air['Ozone'].median())


# In[24]:


A.head()


# In[25]:


A = air['Ozone'].replace(np.NaN,air['Ozone'].mode())


# In[26]:


A.head()


# In[27]:


from sklearn.impute import SimpleImputer 


# In[28]:


from sklearn.impute import SimpleImputer 


# In[29]:


from sklearn.impute import SimpleImputer 


# In[30]:


from sklearn.impute import SimpleImputer 


# In[31]:


from sklearn.impute import SimpleImputer 


# In[32]:


imp = SimpleImputer(missing_values=np.nan,strategy='mean')


# In[33]:


A = imp.fit_transform(air)


# In[34]:


A


# In[35]:


A = pd.DataFrame(A, columns=air.columns)


# In[36]:


A.head()


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


len(A)


# In[39]:


train, test = train_test_split(A)


# In[40]:


len(train)


# In[41]:


len(test)


# In[42]:


train.head()


# In[43]:


train, test = train_test_split(A, test_size = 0.20)


# In[44]:


len(test)


# In[45]:


len(train)


# In[46]:


A.describe()


# In[47]:


from sklearn.preprocessing import StandardScaler


# In[48]:


scaler = StandardScaler()


# In[49]:


B = scaler.fit_transform(A)


# In[50]:


pd.DataFrame(B).describe()


# In[51]:


from sklearn.preprocessing import MinMaxScaler


# In[52]:


scaler = MinMaxScaler()


# In[53]:


B = scaler.fit_transform(A)


# In[54]:


pd.DataFrame(B).describe()


# In[55]:


B = pd.DataFrame(B).describe()


# In[56]:


from sklearn.preprocessing import Binarizer


# In[57]:


bin = Binarizer(threshold=0.5)


# In[58]:


B = bin.fit_transform(B)


# In[59]:


pd.DataFrame(B)


# In[60]:


data=pd.read_csv('student.csv')


# In[61]:


from sklearn.preprocessing import LabelEncoder


# In[62]:


le = LabelEncoder()


# In[63]:


B = le.fit_transform(data['name'])


# In[64]:


B


# In[65]:


B = data[:]


# In[66]:


B['name'] = le.fit_transform(B['name'])


# In[67]:


B


# In[68]:


A


# In[69]:


from sklearn.linear_model import LinearRegression


# In[80]:


X=A['Ozone'].values


# In[81]:


X=X.reshape(-1,1)


# In[82]:


Y = A['Temp']


# In[83]:


model = LinearRegression()


# In[84]:


model.fit(X,Y)


# In[85]:


model.score(X,Y)*100


# 

# In[86]:


model.predict([[128]])


# In[88]:


import matplotlib.pyplot as plt


# In[89]:


plt.scatter(X,Y)


# In[ ]:




