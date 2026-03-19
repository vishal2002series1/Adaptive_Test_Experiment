# Use the official AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Copy the requirements file into the container
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install all the dependencies
RUN pip install -r requirements.txt

# Copy all your Python files into the container
COPY *.py ${LAMBDA_TASK_ROOT}/

# 👉 ADD THIS LINE: Copy the syllabus map into the Lambda root
COPY syllabus_maps.json ${LAMBDA_TASK_ROOT}/

# Set the entrypoint to the lambda_handler function inside app.py
CMD [ "app.lambda_handler" ]