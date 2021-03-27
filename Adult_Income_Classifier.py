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

#reading the file
df = pd.read_csv(r"C:\Users\hp\Downloads\archive\adult.csv")


# In[3]:

#getting an idea of the dataset
df.head()


# In[4]:

#in the above step I found out that there are fields in the dataset where '?' is present. That is why I replaced those fields with nan values
df[df == '?'] = np.nan


# In[5]:

#checking to see if the '?' values have been replaced by nana values
df.head()


# In[6]:

#getting an idea of the number of nan values in each column
df.info()


# In[7]:

#another way of getting an idea of the number of nan values in each column
df.isnull().sum()


# In[8]:

#finding out the mode to fill the nan values with the mode
df['occupation'].describe()


# In[9]:

#fill the nan values with the mode
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

#finding out whether all the nan values have been replaced
df.info()


# In[15]:


rcParams['figure.figsize'] = 12, 12
df[['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']].hist()
#fnding out the distribution of the int values

# In[16]:

#separating the x and y values
dataset = df.drop(['income'], axis=1)

label = df['income']


# In[17]:

#test and train datasets for training the xgboost algorithm
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.3)


# In[18]:

#applying the label encoder for the categorical values
categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])


# In[19]:

#applying the minmaxscaler prior to using the xgboost algorithm
scaler=MinMaxScaler((-1,1))

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = dataset.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = dataset.columns)


# In[20]:


#using the gridsearchcv algorithm in order to find out the best hyperparameters for the xgboost algorithm
param_grid = {'max_depth':[2, 4, 5, 7, 9, 10], 'learning_rate':[0.001, 0.01, 0.1, 0.2, 0.3], 'min_child_weight':[2, 4, 5, 6, 7]}

grid = GridSearchCV(XGBClassifier(), param_grid=param_grid, verbose=3)

grid.fit(dataset, label)


# In[21]:


grid.best_params_


# In[20]:

#fitting the xgboost algorithm to the dataset
model=XGBClassifier()
model.fit(X_train,y_train)


# In[21]:

#running the xgboost algorithm on the test dataset and printing out the accuracy score
y_pred=model.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)


# In[ ]:




