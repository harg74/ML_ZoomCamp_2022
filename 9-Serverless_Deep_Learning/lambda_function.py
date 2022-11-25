import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

#create preprocessor
preprocessor = create_preprocessor('xception', target_size=(299, 299))

interpreter = tflite.Interpreter(model_path='clothing_file.tflite')

#load the weights
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']

output_index = interpreter.get_output_details()[0]['index']

#post predictions
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

#get the data
#url = 'http://bit.ly/mlbookcamp-pants'

def predict(url):
    #we can also load it by path instead by url
    X = preprocessor.from_url(url)

    #make predictions
    interpreter.set_tensor(input_index,X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    return dict(zip(classes, preds[0]))

def lambda_handler(event, context):
    #accessing to value of 'url' key of the dictionary named event
    url = event['url']

    result = predict(url)

    return result

