from flask import Flask,request,rander_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,onehotencoder,LabelEncoder

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return rander_template('index.html')


@app.rout( '/predict',methods=['POST','GET'])
def predict_datapoint():
    if request.method=='GET':
        return rander_template('home.html')
    else:
        pass
    
