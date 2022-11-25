FROM public.ecr.aws/lambda/python:3.8

RUN pip install keras-image-helper
RUN pip pip install --extra-index-url\
    https://google-coral.github.io/py-repo/ tflite_runtime

COPY clothing_file.tflite .
COPY lambda_function.py .

#how to find the lambda function
CMD ["lambda_function.lambda_handler"]