FROM tiangolo/uwsgi-nginx-flask:python3.8

RUN pip install flask-wtf sqlalchemy psycopg2

COPY ./app /app
