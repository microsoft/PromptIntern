# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import requests 
from time import sleep
from requests.exceptions import Timeout
import json 
from tqdm import tqdm
import os
from utils.eval_python import eval_python

sleep_time= 0

def run_llm_direct(model_name,model,messages,temperature=0,max_tokens=3200,top_p=0.95,frequency_penalty=0.0,presence_penalty=0.0,stop=None,n=1):
    if model=="gpt-4":
        URL = ""
        headers = {"api-key": ""} 

    
    if model== "gpt-35-turbo-0125":
        endpoint = ""
        URL = ""
        headers = {"api-key": ""} 

    data = { 
    "messages":messages,  
    "temperature": temperature,
    "max_tokens":max_tokens,
    "top_p":top_p,
    "frequency_penalty":frequency_penalty,
    "presence_penalty":presence_penalty,
    "stop":stop,
    } 
    # print(data)
    # r = requests.post(url = URL, headers=headers, json=data) 
    timeout = 180  
    max_retries = 5  
    retries = 0  
    
    while retries < max_retries:  
        try:  
            r = requests.post(url=URL, headers=headers, json=data, timeout=timeout)  
            break  
        except Timeout:  
            print(f"Request timed out, retrying attempt {retries + 1}...")  
            retries += 1  
    
    # 检查是否请求成功  
    if retries == max_retries:  
        print("Request failed after reaching the maximum number of retries.")  
        raise Timeout
    else:  
        data = r.json()
    data = r.json() 
    
    return data

if __name__ == "__main__":
  
    model_name = ""
    predict_path = f""
    data_path = ''
    output_dir = ""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = output_dir + 'unused.jsonl'
    test_idx = 10
    current = 0

    with open(data_path, "r") as input_file:
        data_lines = input_file.readlines()

    with open(test_path, "r") as input_file, open(output_path, "a") as output_file:
        for i,line in tqdm(enumerate(input_file)):
            if i < current :
                continue

            # if i != 0 and i % 25 == 0: sleep(5)

            tests = ''
            data = json.loads(line)
            messages = data['messages']
            filtered_messages = [message for message in messages if message['role'] != 'assistant']
            response = run_llm_direct(model_name,"gpt-35-turbo-0125",filtered_messages)


            print(response)
            code = response["choices"][0]["message"]["content"]
            # print(code)
            sleep(sleep_time)
            test_data = json.loads(data_lines[test_idx + i])
            tests += '\n'.join(test_data["test_list"])
            eval_data = eval_python(code,tests)
            output_file.write(json.dumps(eval_data) + '\n')