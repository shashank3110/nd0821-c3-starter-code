# Script to train machine learning model.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml.data import process_data,get_data_slices
from ml.model import train_model,inference,compute_model_metrics
# from train_model import process_train_test_data
from joblib import dump,load
from collections import defaultdict
import logging

# Add the necessary imports for the starter code.
import os
print(os.getcwd())

logging.basicConfig(filename='logs/slice_output.txt',encoding='utf-8',
    level=logging.INFO,filemode='w',)

def run(data_path='data/census.csv',cat_features=[],
    label='salary',slicing_feature=None):
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
    #                                         cat_features,label)
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=label, training=True)
    
    

    # eval_data = pd.DataFrame(np.concatenate([X_test,y_test.reshape(-1,1)],axis=1),columns=data.columns)
    
    # # data slicing for evaluation
    # slices, _ = get_data_slices(data=eval_data,slicing_feature=slicing_feature)

    # Train and save a model.
    logging.info("start model training")
    model = train_model(X_train,y_train)
    model_dir = f'model/model.joblib'
    dump(model,model_dir)
    logging.info(f"model save here:{model_dir} ")

    #----adding inference code and check metrics---
    logging.info(f"perform inference on test set with slicing on: {slicing_feature}") 
    model = load(model_dir)

    slice_dict = defaultdict(dict)
    # slice_dict["data"] = slices
    metrics_dict = {}

    for slice_name in data[slicing_feature].unique():
    # for slice_name,data_slice in slice_dict["data"].items():

        logging.info(f"Slice:{slice_name}")
        
        # slice test data on slicing feature value
        data_slice = pd.DataFrame(test,columns=data.columns)
        data_slice = data_slice[data_slice[slicing_feature]==slice_name]

        # Proces the test data with the process_data function.
        X_test, y_test, encoder, lb = process_data(
            data_slice, categorical_features=cat_features, label=label, training=False,
            encoder=encoder,lb=lb)
        

        logging.info(f"Test data shape:{X_test.shape}, Test label shape:{y_test.shape}")
        
        preds = inference(model,X_test)
        
        logging.info("get model_metrics on test set on slice: {slicing_feature}")
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        metrics_dict[slice_name] = {'precision':precision,'recall':recall,
                                       'fbeta':fbeta}
    
        logging.info(f"Slice:{slice_name}: precision={precision},recall={recall},fbeta={fbeta}")
    
    slice_dict["metrics"] = metrics_dict

    return slice_dict

# def run(data_path='data/census.csv',cat_features=[],
#     label='salary',slicing_feature=None):
#     """
#     Function to run end to end :
#     Data loading and processing
#     Training
#     Inference and metrics
#     """
#     # Add code to load in the data.
#     logging.info("read data")
#     data = pd.read_csv(data_path)

#     slices, _ = get_data_slices(data=data,slicing_feature=slicing_feature)
#     slice_dict = defaultdict(dict)
#     slice_dict["data"] = slices
#     model_dict, metrics_dict = {},{}

#     for slice_name,data_slice in slice_dict["data"].items():
#         logging.info(f"Slice:{slice_name}: shape:{data_slice.shape}")
        
#         # Optional enhancement, use K-fold cross validation instead of a train-test split.
#         train, test = train_test_split(data_slice, test_size=0.20)
    
#         # process train/test data
#         logging.info("perform data processing")
#         X_train,y_train,X_test,y_test = process_train_test_data(train,test,
#                                             cat_features,laabel)
        
#         # Train and save a model.
#         logging.info("start model training")
#         model = train_model(X_train,y_train)
#         model_dir = f'model/model_slice_{slice_name}.joblib'
#         dump(model,model_dir)
#         logging.info(f"model save here:{model_dir} ")
#         model_dict[slice_name] = model_dir
         
        
#         #----adding inference code and check metrics---
#         logging.info("perform inference on test set") 
#         model = load(model_dir)
#         preds = inference(model, X_test)
        
#         logging.info("get model_metrics on test set")
#         precision, recall, fbeta = compute_model_metrics(y_test, preds)
#         metrics_dict[slice_name] = {'precision':precision,'recall':recall,
#                                        'fbeta':fbeta}
    
#         logging.info(f"Slice:{slice_name}: precision={precision},recall={recall},fbeta={fbeta}")
#     slice_dict["model"] = model_dict
#     slice_dict["metrics"] = metrics_dict

#     return slice_dict

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
    label = 'salary'
    slicing_feature='education'

    # run script
    results = run(data_path,cat_features,label,slicing_feature)
    logging.info(f"Performance Summary on data slices={results['metrics']}")
