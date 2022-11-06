import bentoml
from bentoml.io import JSON
import numpy as np

from pydantic import BaseModel

class PricePrediction(BaseModel):
    levy: float
    manufacturer: str
    model: str
    prod_year: int
    category: str
    leather_interior: str
    fuel_type: str
    engine_volume: float
    mileage: int
    cylinders: float
    gear_box_type: str
    drive_wheels: str
    doors: int
    wheel: str
    color: str
    airbags: int
    turbo: str

model_ref = bentoml.sklearn.get("car_price_prediction:d4dk7ic4po7ixahg")
dv = model_ref.custom_objects['dictVectorizer']
#runner is the bento mls abstraction for the model itself
#allow us to scale the model separately from the rest
#of the service and we'll get into the different high
#performance scenarios. But for our purposes is  a way
#to access the model and call predict on it
model_runner = model_ref.to_runner()

#pass the runner into the service so that the service knows
#it needs this runner and that model  
svc = bentoml.Service("car-price-prediction", runners=[model_runner])

#create our service endpoint
#the renunner has the same exact signatures as your model did
#in the first place, the only difference is that instead of
#calling the predict directly, you can say predict dot run 

#we'll decorate this endpoint with service.api, which will
#allow us to actually call this endpoint now using rest & curl

@svc.api(input=JSON(pydantic_model=PricePrediction), output=JSON())
def predict(car_price_prediction):
    application_data = car_price_prediction.dict()
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    print(f'log price: {prediction[0]}')
    print(f'actual price: {np.expm1(prediction[0])}')

    result = np.expm1(prediction)[0]

    return result
