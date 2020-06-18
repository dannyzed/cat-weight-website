FROM tiangolo/uwsgi-nginx-flask:python3.8

RUN pip install flask-wtf sqlalchemy psycopg2
RUN pip install scikit-learn matplotlib pandas numpy

COPY ./app /app
