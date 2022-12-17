## Project: Train a model and deploy as an API via FastAPI on Heroku

(This project is a part of my Udacity Nanodegree course project: The skeleton for the project is in this repo. :<br> https://github.com/udacity/nd0821-c3-starter-code)

- This project has CI / CD enabled via GitHub Actions and Heroku

### Environment: python 3.9

### FastAPI 

#### Local API
- Start and run FastAPI app and send get/post requests:
main app script: [starter/main.py](starter/main.py)
```
cd starter
python uvicorn main:app --reload
```
To send a POST request open another command window:
[starter/infer_post_request.py](starter/infer_post_request.py)
```
cd starter
python infer_post_request.py
```

#### Live API
- Note: before sending requests to Live API, make sure the heroku app is deployed and
is live <br> [https://census-data-app1.herokuapp.com/](https://census-data-app1.herokuapp.com/)

To send a POST request to live API:
[infer_live_post_request.py](infer_live_post_request.py)
```
python infer_live_post_request.py 
```

### Training and Data slicing:

- [starter/starter/train_model.py](starter/starter/train_model.py) 
- [starter/starter/train_model_on_slices.py](starter/starter/train_model_on_slices.py)
```
cd starter
python starter/train_model.py
python starter/train_model_on_slices.py 
```

### Data Preprocessing and Model implementation

- [starter/starter/ml/data.py](starter/starter/ml/data.py)
- [starter/starter/ml/model.py](starter/starter/ml/model.py) 

### Dataset, saved models, logs, screenshots:
- Dataset: [starter/data/](starter/data/)
- Model: [starter/model/](starter/model/)
- Screenshots: [starter/screenshots/](starter/screenshots/)
- Logs: [starter/logs/slice_output.txt](starter/logs/slice_output.txt)

### Tests:
- API test script: [starter/test_main.py](starter/test_main.py)
- Model test script: [starter/starter/ml/test_model.py](starter/starter/ml/test_model.py) 
```
cd starter
pytest
```
