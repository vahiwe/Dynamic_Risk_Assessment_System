from flask import Flask, jsonify, request
import pandas as pd
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_values_list, outdated_packages_list
from scoring import score_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    # Get request data in json
    data = request.get_json()

    # Get filename from the post request
    filename = data['filename']

    # Load dataframe from csv
    try:
        df = pd.read_csv(f'{filename}')
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

    # Get the prediction from the model
    prediction = model_predictions(df)

    return jsonify({"prediction": str(prediction)}), 200 #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoringstats():        
    #check the score of the deployed model
    score = score_model()

    return jsonify({"score": score}), 200 #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    #check means, medians, and modes for each column
    summary_stats = dataframe_summary()

    return jsonify({"summary_stats": summary_stats}), 200 #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnosticstats():        
    #check timing and percent NA values

    # get timing statistics
    timing_stats = execution_time()

    # get NA values
    na_stats = missing_values_list()

    # get outdated packages
    outdated_packages = outdated_packages_list()

    return jsonify({"timing_stats": str(timing_stats), "na_stats": str(na_stats), "outdated_packages": str(outdated_packages)}), 200  #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
