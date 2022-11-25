FROM public.ecr.aws/lambda/python:3.8

RUN pip install keras-image-helper
#specific version of tflite compiled for this environment python 3.8 image from aws
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

COPY clothing_file.tflite .
COPY lambda_function.py .

#how to find the lambda function
CMD ["lambda_function.lambda_handler"]