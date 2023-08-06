# pull python base image
FROM python:3.10

# copy application files
ADD ./bank_api /bank_api

# specify working directory
WORKDIR /bank_api/api

# update PIP 
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# expose port for application
EXPOSE 8001

# start Fast api application
CMD ["python", "app/main.py"]
