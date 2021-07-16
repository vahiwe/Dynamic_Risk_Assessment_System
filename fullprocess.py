import ast
import os
import json
import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
input_folder_path = config['input_folder_path']
output_model_path = os.path.join(config['output_model_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Check and read new data
#first, read ingestedfiles.txt
with open(f'{prod_deployment_path}/ingestedfiles.txt', 'r') as f:
    ingested_files = ast.literal_eval(f.read())


#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt

#check for datasets, compile them together, and write to an output file
filenames = os.listdir(os.getcwd() + f'/{input_folder_path}/')

#get only csv files in directory and not in ingested_files
filenames = [filename for filename in filenames if filename.endswith('.csv') and filename not in ingested_files]

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if len(filenames) == 0:
    print('No new data found. Exiting process.')
    exit()

## ingest new data
#if you found new data, ingest it
print('Found new data. Ingesting...')
ingestion.merge_multiple_dataframe()
print('Data ingested. Moving on to training...')

# Create model from newly ingested data
# train model
print('Training model...')
training.train_model()
print('Model trained. Moving on to scoring...')

### Scoring
#score model
print('Scoring model...')
current_model_score = scoring.score_model()
print('Model scored. Moving on to check model drift...')

# read previous model score
with open(f'{prod_deployment_path}/latestscore.txt', 'r') as f:
    previous_model_score = f.read()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
print(current_model_score, previous_model_score)
drift_check = float(current_model_score) > float(previous_model_score)
# check for drift
print('Checking for model drift...')

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if drift_check:
    print("Model drift didn't occur. Exiting process")
    exit()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
print('Model is currently performing better than the previous model. Moving on to deployment...')
# deploy model
print('Deploying model...')
deployment.main()
print('Model deployed. Moving on to diagnostics...')

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

# Run Diagnostics
print('Running diagnostics...')
diagnostics.main()
print('Diagnostics complete. Moving on to reporting...')

# Run Reporting
print('Running reporting...')
reporting.score_model()
print('Reporting complete. Exiting process.')





