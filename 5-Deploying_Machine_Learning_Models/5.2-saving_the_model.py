#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import pickle


# In[2]:


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

#convert column names to lowercase
df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

#convert values of all cols to lowercase
for col in categorical_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')
    
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)
df.churn = (df.churn == 'yes').astype(int)


# In[3]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[4]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = ['gender',
               'seniorcitizen',
               'partner',
               'dependents',
               'phoneservice',
               'multiplelines',
               'internetservice',
               'onlinesecurity',
               'onlinebackup',
               'deviceprotection',
               'techsupport',
               'streamingtv',
               'streamingmovies',
               'contract',
               'paperlessbilling',
               'paymentmethod'
              ]


# In[5]:


def train(df_train, y_train, C):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[6]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    
    return y_pred


# In[7]:


C=1.0
n_splits = 5


# In[8]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)
    
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    
print(f'C = {C}, Mean = {np.round(np.mean(scores),3)}, STD = +-{np.round(np.std(scores),3)}')
    
    


# In[9]:


scores


# In[10]:


y_full_train = df_full_train.churn.values
dv, model = train(df_full_train, y_full_train, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc


# #### Saving the Model 

# In[11]:


output_file = f'model_{C}.bin'
output_file


# In[12]:


f_out = open('output_file', 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


# In[13]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# #### Load the model

# In[14]:


with open(output_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[15]:


dv, model


# In[27]:


customer = {
            'gender': 'female',
            'seniorcitizen': 0,
            'partner': 'yes',
            'dependents': 'no',
            'phoneservice': 'no',
            'multiplelines': 'no',
            'internetservice': 'dsl',
            'onlinesecurity': 'no',
            'onlinebackup': 'yes',
            'deviceprotection': 'no',
            'techsupport': 'no',
            'streamingtv': 'no',
            'streamingmovies': 'no',
            'contract': 'month-to-month',
            'paperlessbilling': 'yes',
            'paymentmethod': 'electronic_check',
            'tenure': 1,
            'monthlycharges': 29.85,
            'totalcharges': 29.85,
        }


# In[28]:


#DictVectorizer expects a list of dictionaries
X = dv.transform([customer])


# In[29]:


model.predict_proba(X)[0,1]


# In[ ]:




