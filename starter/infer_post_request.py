import requests
import json

columns = ["age",	"workclass",	"fnlgt",	"education",	
"education-num",	"marital-status",	"occupation",	
"relationship",	"race",	"sex",	"capital-gain",	"capital-loss",	
"hours-per-week",	"native-country"]

columns = [col.replace('-','_') for col in columns]

values = [31,	"State-gov",	77516,	"Bachelors"	,
13	,"Never-married",	"Adm-clerical",	
"Not-in-family","White","Female",2074, 0,
38,"United-States"]


body = dict(zip(columns,values))

data = json.dumps(body)
r = requests.post("http://127.0.0.1:8000/predict/",data=data)
result = r.json()

print(result)