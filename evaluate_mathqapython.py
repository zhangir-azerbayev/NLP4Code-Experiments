import sys
import re
import random
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

experiment_name = sys.argv[1]
few_shot = int(sys.argv[2])
data_path = sys.arg3[3]
model_path = sys.argv[4]
param_count = sys.argv[5]
device = sys.argv[6]
output_file = sys.argv[7]

model_name = "EleutherAI/gpt-neo-{}".format(param_count)


# maximum token length of text/code, and few_shot_prompt
# Derived from train set
max_length = 474
max_prompt_length = 1387

# Load data 
raw_data = read_mathqapython(data_path)
if few_shot == 1: 
    raw_train_data = read_mathqapython('data/mathqapython_train.json')
    train_size = len(raw_train_data)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


data = MathQAPython(raw_data, tokenizer, max_length)
loader = DataLoader(data, batch_size=1, shuffle=True) 

# Load model 
if few_shot == 1: 
    model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
else: 
    model = GPTNeoForCausalLM.from_pretrained(model_path).to(device)

# Evaluation loop
num_correct = 0 
for batch in tqdm(loader): 
    input_ids, code_sol, answer_sol = batch 

    # Makes few shot prompt if in few-shot regime 
    encoded_few_shot_prompt = tokenizer("", return_tensors="pt")['input_ids']

    if few_shot == 1: 
        while True: 
            idx = random.randrange(train_size) 
            example = "\n\n" + "\n".join([raw_train_data[idx]['text'], 
                raw_train_data[idx]['code'])
            tokenized_example = tokenizer.encode(example, return_tensors="pt")['input_ids']
            longer_encoded_few_shot_prompt = torch.cat([encoded_few_shot_prompt, 
                tokenized_example], axis=1)
            if torch.numel(longer_encoded_few_shot_prompt) <= max_prompt_length:
                encoded_few_shot_prompt = longer_encoded_few_shot_prompt
            else: 
                break 
    
    # Generate outputs
    full_ids = torch.cat([encoded_few_shot_prompt, input_ids], axis=1).to(device)
    generated_ids = model.generate(
        input_ids=full_ids, 
        do_sample=True, 
        temperature=0.4, 
        max_length=2048
    )

    # Isolate one program completion 
    start_idx = encoded_few_shot_prompt.size()[1]
    completion = tokenizer.decode(generated_ids[0, start_idx:])
    program = completion[:re.search('answer.*?\n', completion).span()[1]]
    answer = semisafe_evaluate(program, 'answer', 1)
    if answer is float: 
        if abs(answer - answer_sol) / answer < 0.01: 
            num_correct += 1

    # Writes results to a file
    with open(output_file, "rw") as fle: 
        fle.write("#"*20)
        fle.write("prompt:")
        fle.write(tokenizer.decode(full_ids))
        fle.write("completion: ")
        fle.write(program)
        fle.write("gt completion: ")
        fle.write(code_sol)
        fle.write("answer: ", answer)
        fle.write("label answer: ", answer_sol)

    
accuracy = num_correct / len(data) 

with open(output_file, "rw") as fle: 
    fle.write(accuracy)

print(accuracy)
    









