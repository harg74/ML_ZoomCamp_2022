#!/usr/bin/env python
# coding: utf-8
import  grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from keras_image_helper import create_preprocessor

from flask import Flask
from flask import request
from flask import jsonify

from proto import np_to_protobuf

#where our tf-serving service is running. running in localhost inside docker
host= 'localhost:8500'

#access the previous port, since we are running things locally we can use insecure channel
#since it is running in kubernetes (gateway and tf-serving) tf-serving will not be accessible
#from the outside
channel = grpc.insecure_channel(host)

# we will use this channel to connect our predictions service. This prediction service, 
#is nothing more that tf_serving

#invoke remote services, we will be using for communicating with the actual service. 
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor('xception', target_size=(299, 299))

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

# now we want to send X to prediction service that is running in tf-serving
# we also need to prepare the request, is not JSON format, but protobuf request.
def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name='clothing-model'
    pb_request.model_spec.signature_name='serving_default'
    pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(X))
    return pb_request

def prepare_response(pb_response):
    preds = pb_response.outputs['dense_7'].float_val
    return dict(zip(classes, preds))

def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response

app = Flask('gateway')

@app.route('/predict', methods=['POST'])
def predict_endopint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)

if __name__ =='__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    #url = url = 'http://bit.ly/mlbookcamp-pants'
    #response = predict(url)
    #print(response)