# Use the official AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Copy the requirements file into the container
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install all the dependencies
RUN pip install -r requirements.txt

# Copy all your Python files into the container
COPY schema.py vector_store.py graph.py app.py ${LAMBDA_TASK_ROOT}/

# Set the entrypoint to the lambda_handler function inside app.py
CMD [ "app.lambda_handler" ]