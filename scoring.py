from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # Load Trained model
    with open(f'{output_model_path}/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    # Load test data
    test_data = pd.read_csv(f'{test_data_path}/testdata.csv')

    # Split data
    X = test_data.loc[:,['number_of_employees','lastyear_activity', 'lastmonth_activity']].values.reshape(-1,3)
    y = test_data['exited'].values.reshape(-1,1)

    # Predict
    y_pred = model.predict(X)

    # Calculate F1 score
    f1_score = metrics.f1_score(y, y_pred)

    # Write result to latestscore.txt
    with open(f'{output_model_path}/latestscore.txt', 'w') as file:
        file.write(f'{f1_score}')


if __name__ == '__main__':
    score_model()