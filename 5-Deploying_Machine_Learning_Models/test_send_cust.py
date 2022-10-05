#!/usr/bin/env python
# coding: utf-8

import requests

#url = 'http://0.0.0.0:9696/predict'
url = 'http://127.0.0.1:5000/predict'


customer_id = 'xyz-123'

customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85,
}

response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == True:
    print(f'send promo email to: {customer_id}')
else:
    print(f'not sending promo email to: {customer_id}')
