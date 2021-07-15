import pickle
import os
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path']) 


####################function for deployment
def store_model_into_pickle(model):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    # Storing the model in a pickle file
    pickle.dump(model,open(f'{prod_deployment_path}/trainedmodel.pkl','wb'))

    #copy the latest score.txt file
    score_file_path = os.path.join(output_model_path, "latestscore.txt")
    new_score_file_path = os.path.join(prod_deployment_path, 'latestscore.txt')
    os.rename(score_file_path, new_score_file_path)

    #copy the ingestfiles.txt file
    ingest_file_path = os.path.join(dataset_csv_path, "ingestedfiles.txt")
    new_ingest_file_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    os.rename(ingest_file_path, new_ingest_file_path)

if __name__ == '__main__':
    #load the pickle file
    model = pickle.load(open(os.path.join(output_model_path, "trainedmodel.pkl"), 'rb'))
    store_model_into_pickle(model)
        
        
        

