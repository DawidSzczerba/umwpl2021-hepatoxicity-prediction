#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[2]:


data_train  = pd.read_csv("alt_maccsfp.csv")


# In[3]:


data_train.describe()


# In[4]:


data_train.info()


# In[5]:


sns.distplot(data_train['ALT']);


# In[6]:


data_train.columns[data_train.sum()==0]


# In[7]:


# usunąłem puste kolumny oraz wartości odstające

data_train.drop(data_train.loc[data_train.ALT > 100].index, inplace=True)
data_train.drop(columns=data_train.columns[data_train.sum() == 0], inplace=True)
data_train.reset_index(drop=True, inplace=True)


# In[8]:


data_train


# In[9]:


sns.distplot(data_train['ALT']);


# In[10]:


y = data_train['ALT']
del data_train['ALT']
X = data_train.values
y = y.values

# podzielenie danych na dwa zbiory testowy i treningowy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)


# In[11]:


param_grid = {'alpha':[0.0001,0.001, 0.005, 0.01,0.05,0.1,0.5,1]}


# In[12]:


skf = StratifiedKFold(n_splits=5, random_state=44, shuffle=True)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=44)
# nie dziala :/ 


# In[13]:


kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=666)


# In[14]:


grid_cv_lr = GridSearchCV(Lasso(), param_grid, scoring='r2', cv=kf, return_train_score=True, verbose=1000)
grid_cv_lr_fit = grid_cv_lr.fit(X_train, y_train)


# In[15]:


print(grid_cv_lr_fit.best_score_)
print(grid_cv_lr_fit.best_params_)


# In[16]:


lasso_model = Lasso(alpha =  0.5)
lasso_model.fit(X_train, y_train)


# In[17]:


Y_pred_train = lasso_model.predict(X_train)
print("Accuracy R2 --> ", lasso_model.score(X_train, y_train) * 100)


# In[18]:


Y_pred_test = lasso_model.predict(X_test)
print("Accuracy R2 --> ", lasso_model.score(X_test, y_test) * 100)

