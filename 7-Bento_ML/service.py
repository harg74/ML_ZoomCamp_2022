import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray #cast arrays into nparrys

from pydantic import BaseModel

class CreditApplication(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int

model_ref = bentoml.xgboost.get("credit_risk_model:qnvlhxsqdwqk5ahg")
dv = model_ref.custom_objects['dictVectorizer']
#runner is the bento mls abstraction for the model itself
#allow us to scale the model separately from the rest
#of the service and we'll get into the different high
#performance scenarios. But for our purposes is  a way
#to access the model and call predict on it
model_runner = model_ref.to_runner()

#pass the runner into the service so that the service knows
#it needs this runner and that model  
svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

#create our service endpoint
#the renunner has the same exact signatures as your model did
#in the first place, the only difference is that instead of
#calling the predict directly, you can say predict dot run 

#we'll decorate this endpoint with service.api, which will
#allow us to actually call this endpoint now using rest & curl

@svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON())
async def classify(credit_application):
    application_data = credit_application.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(f'probability: {prediction}')

    result = prediction[0]

    if result > 0.5:
        return result, {'status': 'DECLINED'}

    elif result < 0.25:
        return result, {'status':'MAYBE'}

    else: 
        return result, {'status':'APPROVED'}
