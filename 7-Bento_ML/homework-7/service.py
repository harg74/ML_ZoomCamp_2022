import bentoml
import numpy as np
from bentoml.io import NumpyNdarray #cast arrays into nparrys

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")

model_runner = model_ref.to_runner()

svc = bentoml.Service("mlzoomcamp_homework", runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = model_runner.predict.run(input_series)
    print(f'---> this is the prediction: {result[0]}')
    return result[0]

#model1: qtzdz3slg6mwwdu5
#bentoml build: tag="mlzoomcamp_homework:qhawqfssocegtahg
#docker docker run -it --rm -p 3000:3000 mlzoomcamp_homework:qhawqfssocegtahg serve --production


#model2: jsi67fslz6txydu5
#bento build: tag="mlzoomcamp_homework:qsmxlxcsog6spahg
#docker run -it --rm -p 3000:3000 mlzoomcamp_homework:qsmxlxcsog6spahg serve --production