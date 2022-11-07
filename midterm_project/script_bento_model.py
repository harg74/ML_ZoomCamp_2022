import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score
import bentoml

df = pd.read_csv('resources/train.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns = df.columns.str.lower().str.replace(r'.', '', regex=True)

df = df.drop(['id'], axis=1)
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

#convert values of all cols to lowercase
for col in categorical_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df['mileage'] = df['mileage'].str.lower().str.replace('_km', '')

# split column into multiple columns by delimiter 
df['doors'] = df['doors'].str.split('-', expand=True)[0].str.replace('0', '')
df['doors'] = df['doors'].str.split('-', expand=True)[0].str.replace('>', '')
df['doors'] = df['doors'].astype(int)

df[['engine_volume', 'turbo']] = df['engine_volume'].str.split('_', expand=True)
df['turbo'] = df['turbo'].fillna('normal')
df['engine_volume'] = df['engine_volume'].astype(float)
df['levy'] = df['levy'].replace([r'-'], '', regex=True)
df['levy']= df['levy'].apply(pd.to_numeric)
df['levy']= df['levy'].fillna(0)
df['mileage'] = df['mileage'].astype(int)

df = df.sample(frac=1, random_state=1)

categorical = [
    'manufacturer',
    'model',
    'category',
    'leather_interior',
    'fuel_type',
    'gear_box_type',
    'drive_wheels',
    'wheel',
    'color'
]

numerical = [
    'engine_volume',
    'levy',
    'mileage',
    'prod_year',
    'cylinders',
    'airbags',
    'doors',
]

features = categorical + numerical

sns.histplot(df.price[df.price <100000], bins = 100)
plt.show()


price_logs = np.log1p(df.price)
sns.histplot(price_logs, bins = 100)
plt.show()


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = np.log1p(df_train.price.values)
y_val = np.log1p(df_val.price.values)
y_test = np.log1p(df_test.price.values)

del df_train['price']
del df_val['price']
del df_test['price']

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

val_dicts= df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)


# ### 6.3 Random Forest
rf = RandomForestRegressor(n_estimators=120,
                            random_state=1,
                            max_depth=30,
                            n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)


score = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

y_full_train = np.log1p(df_full_train.price.values)
del df_full_train['price']


full_train_dicts = df_full_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(full_train_dicts)

test_dicts= df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(test_dicts)


rf = RandomForestRegressor(n_estimators=120,
                            random_state=1,
                            max_depth=30,
                            n_jobs=-1)
rf.fit(X_full_train, y_full_train)

y_pred = rf.predict(X_test)


score = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

score, r2


def train(df_train, y_train):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = RandomForestRegressor(n_estimators=190,
                            random_state=1,
                            max_depth=30,
                            n_jobs=-1)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(dicts)
    y_pred = model.predict(X)
    
    return y_pred

dv, model = train(df_full_train, y_full_train)

y_pred = predict(df_test, dv, model)

score = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

score, r2

bentoml.sklearn.save_model("car_price_prediction", model,
                          custom_objects={
                              "dictVectorizer": dv
                          })





