import requests
import json

# API URL
URL = "http://localhost:8080/api/intent?sentence="

# Function to make a prediction using the docker model
def makePrediction(sentence):
    phrase = URL + sentence
    # Making request on the API
    r = requests.get(url = phrase)

    # Extracting data in json format
    data = r.json()
    
    return data

# Make the prediction for all the sentences in input
def makeAllPredictions(data):
    results = [makePrediction(x) for x in data]

    # Classes results by max
    resultsMax = [max(x, key=x.get) for x in results]

    return resultsMax