FROM python:3.8-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD . /app

EXPOSE 5000

RUN python -m pip install -r requirements.txt

# CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "src/frontend:app" ]