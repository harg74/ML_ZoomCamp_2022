import bentoml
from bentoml.io import JSON
import numpy as np
from pydantic import BaseModel

class PricePrediction(BaseModel):
    carat: int 
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

model_ref = bentoml.xgboost.get("diamond_price_prediction:7vyl4ztyhsamfahg")

dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("diamond-price-prediction", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=PricePrediction), output=JSON())

def prediction(diamond_price_prediction):
    application_data = diamond_price_prediction.dict()
    vector=dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    print(f'Diamond price: {np.expm1(prediction)}')

    result = prediction[0]

    return np.expm1(result)

