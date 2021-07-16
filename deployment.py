import os
import json
import shutil

# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path'])


# function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the
    # ingestfiles.txt file into the deployment directory

    # copy the latest pickle file
    latest_pickle_file = os.path.join(output_model_path, "trainedmodel.pkl")
    new_pickle_file_path = os.path.join(
        prod_deployment_path, "trainedmodel.pkl")
    shutil.copy2(latest_pickle_file, new_pickle_file_path)

    # copy the latest score.txt file
    score_file_path = os.path.join(output_model_path, "latestscore.txt")
    new_score_file_path = os.path.join(prod_deployment_path, 'latestscore.txt')
    shutil.copy2(score_file_path, new_score_file_path)

    # copy the ingestfiles.txt file
    ingest_file_path = os.path.join(dataset_csv_path, "ingestedfiles.txt")
    new_ingest_file_path = os.path.join(
        prod_deployment_path, 'ingestedfiles.txt')
    shutil.copy2(ingest_file_path, new_ingest_file_path)


def main():
    # store the model into the pickle file
    store_model_into_pickle()


if __name__ == '__main__':
    main()
