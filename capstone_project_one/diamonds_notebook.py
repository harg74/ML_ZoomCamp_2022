
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import xgboost as xgb

data = pd.read_csv('resources/diamonds.csv')
data.head()

df = data.drop(['Unnamed: 0'], axis=1)

categorical = [
    'cut',
    'color',
    'clarity'
]

numerical = [
    'carat',
    'depth',
    'table',
    'x',
    'y',
    'z',
]

features=numerical + categorical

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

#full_train & test

y_full_train = np.log1p(df_full_train.price.values)
del df_full_train['price']

full_train_dicts = df_full_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(full_train_dicts)

y_test = np.log1p(df_test.price.values)
del df_test['price']

test_dicts= df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(test_dicts)

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

watchlist = [(dfulltrain, 'full_train'), (dtest, 'test')]

def train(df_train, y_train):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    dfulltrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    
    xgb_params = {
        'eta': 0.2, 
        'max_depth': 10,
        'min_child_weight': 10,

        'objective': 'reg:squarederror',
        'nthread': 8,

        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.train(xgb_params, dfulltrain,
                      evals=watchlist,
                      num_boost_round=100)
    return dv, model


def predict(df, dv, model, y_val):
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    X_val = dv.transform(dicts)
    
    dtest = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    
    y_pred = model.predict(dtest)

    score = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f'Diamond, price: {np.expm1(y_pred)}, RMSE Score: {score}, R2: {r2}')

    return y_pred


dv, model = train(df_full_train, y_full_train)


y_pred = predict(df_test, dv, model, y_test)

