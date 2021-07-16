import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    # create empty dataframe object
    final_dataframe = pd.DataFrame(columns=["corporation","lastmonth_activity","lastyear_activity","number_of_employees","exited"])

    # Ingested files
    ingested_files = []

    #check for datasets, compile them together, and write to an output file
    filenames = os.listdir(os.getcwd() + f'/{input_folder_path}/')

    #get only csv files in directory
    filenames = [filename for filename in filenames if filename.endswith('.csv')]

    # Open file to record details
    with open(f'{output_folder_path}/ingestedfiles.txt', 'w') as f:

        # Read files into dataframes
        for each_filename in filenames:
            currentdf = pd.read_csv(os.getcwd() + f'/{input_folder_path}/' + each_filename)
            final_dataframe = final_dataframe.append(currentdf).reset_index(drop=True)
            ingested_files.append(each_filename)
            
        # Store ingested files in a text file
        f.write(str(ingested_files))
        
        # Deduplicate dataframe
        final_dataframe = final_dataframe.drop_duplicates()

        # Write dataframe to csv    
        final_dataframe.to_csv(f'{output_folder_path}/finaldata.csv', index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()
