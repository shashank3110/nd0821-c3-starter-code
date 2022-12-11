# unit test script for data loading, training, inference
from .model import train_model,inference,compute_model_metrics
from .data import process_data
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
import logging

logging.basicConfig(level=logging.INFO) 

data = pd.read_csv('data/census.csv')
train, test = train_test_split(data,test_size=0.20)

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

def test_train_model():
    """
    Unit test for train_model()
    """
    logging.info("Testing train_model()")
    X_train, y_train, encoder, lb = process_data(train,training=True,
        label='salary',categorical_features=cat_features)
    pytest.encoder = encoder
    pytest.lb = lb

    model = train_model(X_train,y_train)
    pytest.model = model
    model_dir = 'model/test_model.joblib'
    logging.info(f"test model will be saved here={model_dir}")
    joblib.dump(model,model_dir)
    assert isinstance(model,GradientBoostingClassifier)


def test_inference():
    """
    Unit test for inference()
    """
    logging.info("Testing inference()")
    X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=pytest.encoder,lb=pytest.lb)

    preds = inference(pytest.model, X_test)
    pytest.preds = preds
    pytest.y_test = y_test
    assert preds.shape == y_test.shape


def test_compute_model_metrics():
    """
    Unit test for compute_model_metrics()
    """
    logging.info("Testing compute_model_metrics()")
    precision, recall, fbeta = compute_model_metrics(pytest.y_test,pytest.preds)
    logging.info(f"precision={precision},recall={recall},fbeta={fbeta}")

    assert precision >= 0 and precision <= 1.0
    assert recall >= 0 and recall <= 1.0
    assert fbeta >= 0 and fbeta <= 1.0
