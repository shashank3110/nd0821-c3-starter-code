# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model,inference,compute_model_metrics
from joblib import dump,load
import logging

# Add the necessary imports for the starter code.
import os
print(os.getcwd())

logging.basicConfig(filename='logs/output.txt',encoding='utf-8',
    level=logging.INFO,filemode='w')



def run(data_path='data/census.csv',cat_features=[],label="salary"):
    """
    Function to run end to end :
    Data loading and processing
    Training
    Inference and metrics
    """
    # Add code to load in the data.
    logging.info("read data")
    data = pd.read_csv(data_path)


    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data,random_state=42, test_size=0.20)
    
    # process train/test data
    logging.info("perform data processing")
    # X_train,y_train,X_test,y_test = process_train_test_data(train,test,
                                        # cat_features,label,)

    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=label, training=True)
    
    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label=label, training=False,
    encoder=encoder,lb=lb)

    dump(encoder,'model/encoder.joblib')
    dump(lb,'model/lb.joblib')

    # Train and save a model.
    logging.info("start model training")
    model = train_model(X_train,y_train)
    model_dir = 'model/model.joblib'
    dump(model,model_dir)
    logging.info(f"model save here:{model_dir} ")
    
    #----adding inference code and check metrics---
    logging.info("perform inference on test set") 
    model = load(model_dir)
    preds = inference(model, X_test)
    
    logging.info("get model_metrics on test set")
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logging.info(f"precision={precision},recall={recall},fbeta={fbeta}")


if __name__=='__main__':

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",]

    data_path='data/census.csv'
    label = "salary"

    # run script
    run(data_path,cat_features,label)
