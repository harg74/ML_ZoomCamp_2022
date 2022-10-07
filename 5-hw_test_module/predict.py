import pickle
from flask import request, Flask, jsonify

model_file = 'model1.bin'
model_file2 = 'model2.bin'
dv_file = 'dv.bin'

#with open(model_file, 'rb') as f_in:
#    dv, model = pickle.load(f_in)

with open(model_file2, 'rb') as model_in:
    model = pickle.load(model_in)
with open(dv_file, 'rb') as dv_in:
    dv = pickle.load(dv_in)

app=Flask('churn')

@app.route('/predict', methods=["POST"])
def predict():
    #get_json: tell flask we are getting json and extract the body of request as json
    #requests: will return the body of the request as python dictionary
    #it will take the body of the request, assume that is json, parse it and return a python dict.
    customer = request.get_json()

    #DictVectorizer expects a list of dictionaries
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:,1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred), #np float into python float
        'churn': bool(churn) #np boolean into python boolean
    }

    #we return a dictionary that will be converted into a json with jsonify
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
    #app.run(debug=True, host='0.0.0.0', port=9696)



