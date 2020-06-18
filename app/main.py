from wtforms import Form, BooleanField, StringField, PasswordField, validators
from flask import request, render_template, flash, Flask
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from app.gp import make_plots
import asyncio

loop = asyncio.get_event_loop()

if 'FLASK_DEBUG' in os.environ:
    SQL_TABLE = "weight_development"
else:
    SQL_TABLE = "weight"

MAX_RETRY = 5

Base = declarative_base()

db = sqlalchemy.create_engine(r'postgresql+psycopg2://{}:{}@{}:5432/weight'.format(os.environ['SQL_USER'], os.environ['SQL_PASS'], os.environ['SQL_SERVER']))
Session = sessionmaker(bind=db)

db_session = Session()

def run_query(f, attempts=5):
    while attempts > 0:
        attempts -= 1
        try:
            return f() # "break" if query was successful and return any results
        except sqlalchemy.exc.DBAPIError as exc:
            if attempts > 0 and exc.connection_invalidated:
                db_session.rollback()
            else:
                raise

class WeightEntry(Base):
    __tablename__ = SQL_TABLE
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    name = sqlalchemy.Column(sqlalchemy.String)
    weight = sqlalchemy.Column(sqlalchemy.Float)
    time = sqlalchemy.Column(sqlalchemy.DateTime)

app = Flask(__name__)

app.secret_key = os.environ['FLASK_SECRET_KEY']

class WeightForm(Form):
    name = StringField('Name', [validators.DataRequired()])
    weight = StringField('Weight', [validators.DataRequired()])

@app.route('/submit', methods=['GET', 'POST'])
def register():
    try:
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        valid = False
        if ip.startswith('127.0.0.1') or ip.startswith('192.168'):
            valid = True
        if not valid:
            return "Error"
        form = WeightForm(request.form)

        def query_fn():
            return db_session.query(WeightEntry).order_by(WeightEntry.time.desc()).limit(10)
        
        data = run_query(query_fn)

        if request.method == 'POST' and form.validate():
            weight = WeightEntry(name=form.name.data, weight=form.weight.data, time=datetime.now())

            def commit_fn():
                db_session.add(weight)
                db_session.commit()
            run_query(commit_fn)
            return render_template('weight.html', form=form, data=data)
        return render_template('weight.html', form=form, data=data)
    except Exception as e:
        db_session.rollback()
        return str(e)

@app.route('/')
def home():
    return render_template("dashboard.html")

@app.route('/plot')
def plot():
    loop.run_until_complete(make_plots())
    return 'Done'