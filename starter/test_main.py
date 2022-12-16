"""
Local api test case script
for main.py
"""

from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)


def test_get_greeting():
    """
    Test get request
    """
    res = client.get('/')

    assert res.status_code == 200
    assert res.json() == {"message":"Welcome to the Inference portal"}

def test_infer():
    """
    Test post request with data.
    """
    # case 1
    body1 = {
    "age": 28,
    "workclass": "Private",	
    "fnlgt": 58106,
    "education": "Masters",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "Asian",
    "sex": "Male",
    "capital_gain": 8000,
    "capital_loss": 0,
    "hours_per_week": 42,
    "native_country": "India"
    }


    data1 = json.dumps(body1)

    res1 = client.post('/predict/',data=data1)

    assert res1.status_code == 200
    assert res1.json() == '{"results": [">50K"]}' #match json output as string

    # case 2
    body2 = {
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
    "capital_gain": 2740,
    "capital_loss": 0,
    "hours_per_week": 42,
    "native_country": "United-States"
    }

    data2 = json.dumps(body2)

    res2 = client.post('/predict/',data=data2)

    assert res2.status_code == 200
    assert res2.json() == '{"results": ["<=50K"]}' #match json output as string

def test_infer_no_data():
    """
    Test post request with empty data,
    this should fail!!
    """
    data = json.dumps({})

    res = client.post('/predict/',data=data)

    assert res.status_code != 200 # fil as there is no data in post request






