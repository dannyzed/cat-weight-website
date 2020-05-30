from wtforms import Form, BooleanField, StringField, PasswordField, validators
from flask import request, render_template, flash, Flask
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

db = sqlalchemy.create_engine(r'postgresql+psycopg2://{}:{}@{}:5432/weight'.format(os.environ['SQL_USER'], os.environ['SQL_PASS'], os.environ['SQL_SERVER']))
Session = sessionmaker(bind=db)

db_session = Session()

class WeightEntry(Base):
    __tablename__ = 'weight'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    name = sqlalchemy.Column(sqlalchemy.String)
    weight = sqlalchemy.Column(sqlalchemy.Float)
    time = sqlalchemy.Column(sqlalchemy.DateTime)

app = Flask(__name__)

app.secret_key = os.environ['FLASK_SECRET_KEY']

class WeightForm(Form):
    name = StringField('Name', [validators.DataRequired()])
    weight = StringField('Weight', [validators.DataRequired()])

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = WeightForm(request.form)
    data = db_session.query(WeightEntry).order_by(WeightEntry.time.desc()).limit(10)
    if request.method == 'POST' and form.validate():
        weight = WeightEntry(name=form.name.data, weight=form.weight.data, time=datetime.now())
        db_session.add(weight)
        db_session.commit()
        return render_template('weight.html', form=form, data=data)
    return render_template('weight.html', form=form, data=data)