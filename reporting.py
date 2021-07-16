import pickle
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


# Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace

    # Load Trained model
    with open(f'{output_model_path}/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    # Load test data
    test_data = pd.read_csv(f'{test_data_path}/testdata.csv')

    # Split data
    X = test_data.loc[:,
                      ['number_of_employees',
                       'lastyear_activity',
                       'lastmonth_activity']].values.reshape(-1,
                                                             3)
    y = test_data['exited'].values.reshape(-1, 1)

    # Predict
    y_pred = model.predict(X)

    # Calculate Confusion Matrix
    cm = metrics.confusion_matrix(y, y_pred)

    # Create a confusion matrix plot
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".3f",
        linewidths=.5,
        square=True,
        cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # Save plot
    plt.savefig(f'{output_model_path}/confusionmatrix.png')


if __name__ == '__main__':
    score_model()
