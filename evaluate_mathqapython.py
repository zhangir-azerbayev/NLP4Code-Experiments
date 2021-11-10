import sys
import re
import random
import yaml
import json
import os
import itertools
from tqdm import tqdm 
random.seed(1)

import numpy as np 
np.random.seed(1)

import torch
from torch.utils.data import DataLoader
torch.manual_seed(1)

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from dataloader import read_mathqapython, MathQAPython 

from execute_code import semisafe_evaluate

to_dump = ""

config_path = sys.argv[1]

with open(config_path, "r") as f: 
    cfg = yaml.safe_load(f) 

experiment_name = cfg['experiment_name']
few_shot = cfg['few_shot']
data_path = cfg['data_path']
model_path = cfg['model_path']
param_count = cfg['param_count']
device = cfg['device']
pass_at = cfg['pass_at']
temp = cfg['temp']

os.mkdir("results/" + experiment_name)

model_name = "EleutherAI/gpt-neo-{}".format(param_count)


# maximum token length of text/code, and few_shot_prompt
# max_length = 256 works for almost everything 
max_length = 512
max_prompt_length = 1387 # not currently in use

# Load data 
print("loading data")
raw_data = read_mathqapython(data_path)
if few_shot == 1: 
    raw_train_data = read_mathqapython('data/mathqapython_train.json')
    train_size = len(raw_train_data)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


data = MathQAPython(raw_data, tokenizer, max_length)
loader = DataLoader(data, batch_size=1, shuffle=True) 

print("loading model")
# Load model 
if few_shot == 1: 
    model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
else: 
    model = GPTNeoForCausalLM.from_pretrained(model_path).to(device)

# Evaluation loop
print("doing evaluation loop")
num_correct = 0 
no_errors = 0 
sum_passk = 0 
for batch in tqdm(loader): 
    input_ids, code_sol, mask, answer_sol = batch 

    # Removes padding tokens 
    input_ids = torch.unsqueeze(input_ids[~(input_ids==tokenizer.eos_token_id)], 0)

    encoded_few_shot_prompt = tokenizer("", return_tensors="pt")['input_ids']

    # Makes few shot prompt if in few-shot regime 
    if few_shot == 1: 
        idxs = random.sample(range(train_size), 3)
        few_shot_prompt = "\n\n".join([raw_train_data[idx]['text'] + "\n" + 
            raw_train_data[idx]['code'] for idx in idxs]) + "\n\n"
        encoded_few_shot_prompt = tokenizer.encode(few_shot_prompt, 
                return_tensors="pt")
    
    # Generate outputs
    # Setting max_new_tokens=256 captures all but ~10 training examples
    full_ids = torch.cat([encoded_few_shot_prompt, input_ids], axis=1).to(device)
    with torch.no_grad(): 
        generated_ids = model.generate(
            input_ids=full_ids.long(), 
            do_sample=True, 
            temperature=temp, 
            max_new_tokens=256, 
            pad_token_id=tokenizer.eos_token_id, 
            num_return_sequences=pass_at
        )

    # Does pass@k
    correct_per_sample = 0 
    correct_program = None 
    start_idx = encoded_few_shot_prompt.size()[1]
    
    
    for sample in generated_ids: 
        completion = tokenizer.decode(sample[start_idx:], skip_special_tokens=True)
        answer_locs = re.search('^answer.*?\n', completion)
        if answer_locs: 
            program = completion[:answer_locs.span()[1]]
        else: 
            program = completion 
        answer = semisafe_evaluate(program, 'answer', 1)
        if isinstance(answer, float): 
            if abs((answer - answer_sol) / answer) < 0.01: 
                correct_per_sample += 1
                correct_program = program 
                answer_to_save = answer 
            no_errors += 1

    if correct_program: 
        program_to_save = correct_program
        num_correct += 1
    else: 
        program_to_save = program # Effectively, just a sample 
        answer_to_save = answer 

    passk = float(correct_per_sample)/pass_at 
    sum_passk += passk 

    # Writes results to a file
    to_dump += "\n" + "#"*20 + "\n"
    to_dump += "PROMPT: \n"
    to_dump += tokenizer.decode(full_ids.squeeze(), skip_special_tokens=True)
    to_dump += "\n\nGENERATED COMPLETION: \n" 
    to_dump += program_to_save 
    to_dump += "\n\nLABEL COMPLETION:\n"
    to_dump += tokenizer.decode(code_sol.squeeze(), skip_special_tokens=True)
    to_dump += "\n\nANSWER: " + str(answer_to_save) + "\n"
    to_dump += "\nLABEL ANSWER: " + str(answer_sol.item()) 
    to_dump += "\n\nPASS@1: " + str(float(correct_per_sample)/pass_at) + "\n"



    
accuracy = num_correct / len(data) 
execution_rate = no_errors / (len(data)*pass_at)
avg_passk = sum_passk/len(data)

metrics = {
        "pass@k": accuracy, 
        "execution_rate": execution_rate, 
        "avg_pass@1": avg_passk
        }

with open("results/" + experiment_name + "/metrics.txt", "w") as fle: 
    fle.write(json.dumps(metrics, indent=2, sort_keys=True))

with open("results/" + experiment_name + "/inferences.txt", "w") as fle: 
    fle.write(to_dump)

with open("results/" + experiment_name + "/config.yml", "w") as fle: 
    yaml.dump(cfg, fle)


print(accuracy)
