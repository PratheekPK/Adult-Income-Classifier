#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import rcParams


# In[2]:


df = pd.read_csv(r"C:\Users\hp\Downloads\archive\adult.csv")


# In[3]:


df.head()


# In[4]:


df[df == '?'] = np.nan


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df['occupation'].describe()


# In[9]:


df['occupation'] = df['occupation'].fillna('Prof-specialty')


# In[10]:


df['workclass'].describe()


# In[11]:


df['workclass'] = df['workclass'].fillna('Private')


# In[12]:


df['native.country'].describe()


# In[13]:


df['native.country'] = df['native.country'].fillna('United-States')


# In[14]:


df.info()


# In[15]:


rcParams['figure.figsize'] = 12, 12
df[['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']].hist()


# In[16]:


dataset = df.drop(['income'], axis=1)

label = df['income']


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.3)


# In[18]:


categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])


# In[19]:


scaler=MinMaxScaler((-1,1))

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = dataset.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = dataset.columns)


# In[20]:



param_grid = {'max_depth':[2, 4, 5, 7, 9, 10], 'learning_rate':[0.001, 0.01, 0.1, 0.2, 0.3], 'min_child_weight':[2, 4, 5, 6, 7]}

grid = GridSearchCV(XGBClassifier(), param_grid=param_grid, verbose=3)

grid.fit(dataset, label)


# In[21]:


grid.best_params_


# In[20]:


model=XGBClassifier()
model.fit(X_train,y_train)


# In[21]:


y_pred=model.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)


# In[ ]:




