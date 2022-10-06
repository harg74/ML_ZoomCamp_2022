#!/usr/bin/env python
# coding: utf-8

import requests

#url = 'http://0.0.0.0:9696/predict'
#this local hos was used when we are not using EB, this is for test locally
#url = 'http://127.0.0.1:5000/predict'

host = 'churn-serving-env.eba-igvcyijm.us-east-1.elasticbeanstalk.com'

#we dont have to specify the port, since has implicit port 80
#port 80 is the port that the server "listens to" or expects to receive from a Web client
#EB will also internally map it to the churn service port 9696
url = f'http://{host}/predict'


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
