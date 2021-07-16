import pandas as pd
import numpy as np
import timeit
import subprocess
import os
import json
import pickle

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path']) 

##################Function to get model predictions
def model_predictions(dataframe):
    #read the deployed model and a test dataset, calculate predictions

    # Load Trained model
    with open(f'{prod_deployment_path}/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    # Split data
    X = dataframe.loc[:,['number_of_employees','lastyear_activity', 'lastmonth_activity']].values.reshape(-1,3)
    y = dataframe['exited'].values.reshape(-1,1)

    # Predict
    y_pred = model.predict(X)

    return y_pred#return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here

    ################### Load dataframe for summarization
    dataset = pd.read_csv(f'{dataset_csv_path}/finaldata.csv')

    # Get mean of numeric columns in dataframe
    themeans = list(dataset.mean())

    # Get median of numeric columns in dataframe
    themedians = list(dataset.median())

    # Get stdev of numeric columns in dataframe
    thestandardevs = list(dataset.std())

    return [themeans, themedians, thestandardevs] #return value should be a list containing all summary statistics

##################Function to check for any missing values
def missing_values_list():
    ################### Load dataframe to check for missing values
    dataset = pd.read_csv(f'{dataset_csv_path}/finaldata.csv')

    ################### Check for missing values
    nas=list(dataset.isna().sum())

    # Get percentage of missing values
    napercents = [nas[i] / len(dataset.index) for i in range(len(nas))]

    return napercents #return value should be a list containing all missing values

########### Calculate time for ingestion
def ingestion_timing():
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing=timeit.default_timer() - starttime
    return timing

############ Calculate time for training
def training_timing():
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing=timeit.default_timer() - starttime
    return timing

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    time_record = []

    # Get time for ingestion
    ingestion_time = ingestion_timing()
    time_record.append(ingestion_time)

    # Get time for training
    training_time = training_timing()
    time_record.append(training_time)

    return time_record#return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    outdated = subprocess.check_output(['pip', 'list','--outdated'])
    with open('outdated.txt', 'wb') as f:
        f.write(outdated)

    return outdated


def main():
    ###################Load test data
    test_data = pd.read_csv(f'{test_data_path}/testdata.csv')

    ###################Call model predictions
    model_predictions(test_data)

    ###################Call summary statistics
    dataframe_summary()

    ###################Call missing values
    missing_values_list()

    ###################Call timings
    execution_time()

    ###################Call dependencies
    outdated_packages_list()


if __name__ == '__main__':
    main()