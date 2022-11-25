#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('autosave', '0')


# In[2]:


import tensorflow as tf
from tensorflow import keras
import numpy as np


# In[3]:


#download model and save it as "clothing-model.h5"

#!wget https://github.com/alexeygrigorev/mlbookcamp-code/releases/download/chapter7-model/xception_v4_large_08_0.894.h5 -O clothing-model.h5


# In[4]:


#download pants image
#!wget http://bit.ly/mlbookcamp-pants -O pants.jpg


# In[5]:


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input


# In[6]:


model = keras.models.load_model('clothing-model.h5')


# In[7]:


img = load_img('pants.jpg', target_size=(299,299))
x = np.array(img)
X = np.array([x])

X = preprocess_input(X)


# In[8]:


X.shape


# In[9]:


pred = model.predict(X)


# In[10]:


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


# In[11]:


dict(zip(classes, pred[0]))


# ## Convert Keras to TF-Lite

# In[12]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('clothing_file.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


# In[13]:


import tensorflow.lite as tflite


# In[14]:


#load the model
interpreter = tflite.Interpreter(model_path='clothing_file.tflite')

#load the weights
interpreter.allocate_tensors()


# In[15]:


#get the inputs
#the index says which part of the model is the input, where the input should go.
interpreter.get_input_details()


# In[16]:


input_index = interpreter.get_input_details()[0]['index']


# In[17]:


interpreter.get_output_details()


# In[18]:


# we have index 0 again because we have only one output, sometimes we can have more
#it means we have 229 things inside our model and this is the last one.
output_index = interpreter.get_output_details()[0]['index']


# In[19]:


#initialize the input of the interpreter with X (pants image)
interpreter.set_tensor(input_index, X)


# In[20]:


#invoke all the combinations in the neural network. We got the input data and we passed though all this
#base model, layers of the neural network and we now have the results sitting in the 
#output index sitting waiting for us
interpreter.invoke()


# In[21]:


#we get our predictions
preds = interpreter.get_tensor(output_index)


# In[22]:


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


# In[23]:


dict(zip(classes, pred[0]))


# ## Removing TF dependency 
# 
# We won't have dependency on tensorflow, we foound how keras implement this and took this code and use it here.

# In[24]:


from PIL import Image


# In[25]:


with Image.open('pants.jpg') as img:
    img = img.resize((299, 299), Image.NEAREST)


# In[26]:


def preprocess_input(x):
    x/=127.5
    x-=1.
    return x


# In[27]:


x = np.array(img, dtype='float32')
X = np.array([x])

X = preprocess_input(X)


# In[28]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[29]:


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


# In[30]:


dict(zip(classes, pred[0]))


# ## Simpler way  of doing it with keras image helper

# In[34]:


pip install keras-image-helper


# In[35]:


#by installing tflite we avoid relying on tensorflow
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime


# In[36]:


import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


# In[ ]:


#load the model
interpreter = tflite.Interpreter(model_path='clothing_file.tflite')

#load the weights
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']

output_index = interpreter.get_output_details()[0]['index']


# In[ ]:


#create preprocessor
preprocessor = create_preprocessor('xception', target_size=(299, 299))

#get the data
url = 'http://bit.ly/mlbookcamp-pants'

#we can also load it by path instead by url
X = preprocessor.from_url(url)


# In[ ]:


#make predictions
interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[ ]:


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

dict(zip(classes, pred[0]))


# In[ ]:




