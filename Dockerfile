FROM tiangolo/uvicorn-gunicorn:python3.8-slim

WORKDIR /app

ADD requirements.txt .
RUN pip install -r requirements.txt

COPY ./model_deployment /model_deployment
COPY .app.py .
COPY ./autonlp /autonlp

#docker build -t autonlp .
#docker run -d --name apiautonlpcontainer -p 80:80 autonlp