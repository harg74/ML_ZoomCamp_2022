import requests

#url='http://localhost:9696/predict'
#url='http://localhost:8080/predict'
url='http://af85d63246e44456a81c31d019d01cb1-2132056441.us-east-1.elb.amazonaws.com/predict'

data = {'url':'http://bit.ly/mlbookcamp-pants'}

result = requests.post(url, json=data).json()
print(result)