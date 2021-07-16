import requests
import os
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path']) 

#Call each API endpoint and store the responses
# call prediction endpoint
response1 = requests.post(f'{URL}prediction', json={"filename":"testdata/testdata.csv"})
response2 = requests.get(f'{URL}scoring')
response3 = requests.get(f'{URL}summarystats')
response4 = requests.get(f'{URL}diagnostics')

# #combine all API responses
responses = [response1, response2, response3, response4]

# #write the responses to your workspace
with open(f'{output_model_path}/apireturns.txt', 'w') as f:
    for response in responses:
        f.write(response.text)



