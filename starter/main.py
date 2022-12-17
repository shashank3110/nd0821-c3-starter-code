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
import os


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

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",	
                "fnlgt": 77516,
                "education": "Masters",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 20740,
                "capital_loss": 0,
                "hours_per_week": 42,
                "native_country": "United-States"
            }
        }


    

@app.get("/")
async def greeting():
    return {"message": "Welcome to the Inference portal"}


@app.post("/predict/")
async def infer(body: Data):# use of type hints for body i.e. Data
    
    if os.path.exists('model/'):
        base_path = 'model'
    else:
        base_path = os.path.join(os.getcwd(),'starter/model') # run main.py from outside starter.s
    
    model = joblib.load(f'{base_path}/model.joblib')
    encoder = joblib.load(f'{base_path}/encoder.joblib')
    lb = joblib.load(f'{base_path}/lb.joblib')
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
