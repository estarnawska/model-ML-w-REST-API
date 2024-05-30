From python:3.11-slim-buster

WORKDIR /main

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY model.pkl .

COPY app.py .

ENV FLASK_APP=app

EXPOSE 5050

CMD ["python", "app.py"]
