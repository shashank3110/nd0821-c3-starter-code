"""
API for model inference.
"""


# Put the code for your API here.
# from starter.train_model import *
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.model import inference
from starter.ml.data import process_data
import requests
import joblib
import json
import pandas as pd
import numpy as np

app = FastAPI()

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
] 


class Data(BaseModel):

    age                :  int
    workclass         :  str
    fnlgt              :  int
    education         :  str
    education_num      :  int
    marital_status    :  str
    occupation        :  str
    relationship      :  str
    race              :  str
    sex               :  str
    capital_gain       :  int
    capital_loss       :  int
    hours_per_week     :  int
    native_country    :  str

    

@app.get("/")
async def greeting():
    return {"message": "Welcome to the Inference portal"}


@app.post("/predict/")
async def infer(body: Data):# use of type hints for body i.e. Data
    model = joblib.load('model/model.joblib')
    encoder = joblib.load('model/encoder.joblib')
    lb = joblib.load('model/lb.joblib')
    body = json.loads(body.json())
    X = np.array(list(body.values()))
    cols = [col.replace('_','-') for col in body.keys()]
    X = pd.DataFrame(X.reshape(1,-1),columns=cols)
    
    # process input data for inference
    X_categorical = X[cat_features].values
    X_continuous = X.drop(*[cat_features], axis=1)
    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)

    # X,_,_,_ = process_data(body.values,categorical_features=cat_features, 
    # label=None, training=False, encoder=encoder, lb=lb)
    
    # perform inference
    preds = inference(model, X)

    # use label binarizer inverse_transaform to return the predicted label.
    return json.dumps({'results':lb.inverse_transform(preds).tolist()})
