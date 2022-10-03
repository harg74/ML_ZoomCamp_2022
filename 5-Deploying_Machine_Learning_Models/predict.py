import pickle

with open('output_file', 'rb') as f_in:
    dv, model = pickle.load(f_in)

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


#DictVectorizer expects a list of dictionaries
X = dv.transform([customer])

y_pred = model.predict_proba(X)[0,1]

print('input', customer)
print('churn probability', y_pred)