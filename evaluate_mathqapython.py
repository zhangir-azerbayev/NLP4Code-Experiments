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

to_dump = ""

experiment_name = sys.argv[1]
few_shot = int(sys.argv[2])
data_path = sys.argv[3]
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
for batch in tqdm(loader): 
    input_ids, mask, code_sol, answer_sol = batch 

    # Removes padding tokens 
    input_ids = torch.unsqueeze(input_ids[~(input_ids==tokenizer.eos_token_id)], 0)

    encoded_few_shot_prompt = tokenizer("", return_tensors="pt")['input_ids']

    # Makes few shot prompt if in few-shot regime 
    if few_shot == 1: 
        idxs = random.sample(range(train_size), 2)
        few_shot_prompt = "\n\n".join([raw_train_data[idx]['text'] + "\n" + 
            raw_train_data[idx]['code'] for idx in idxs]) + "\n\n"
        encoded_few_shot_prompt = tokenizer.encode(few_shot_prompt, 
                return_tensors="pt")
    
    # comment previous and uncomment this block to make 
    # maximally long few-shot prompts
    # 
    # if few_shot == 1: 
    #     while True: 
    #         idx = random.randrange(train_size) 
    #         example = "\n".join([raw_train_data[idx]['text'], 
    #             raw_train_data[idx]['code']]) + "\n\n"
    #         tokenized_example = tokenizer(example, return_tensors="pt")['input_ids']
    #         longer_encoded_few_shot_prompt = torch.cat([encoded_few_shot_prompt, 
    #             tokenized_example], axis=1)
    #         if torch.numel(longer_encoded_few_shot_prompt) <= max_prompt_length:
    #             encoded_few_shot_prompt = longer_encoded_few_shot_prompt
    #         else: 
    #             break 
    
    # Generate outputs
    # Setting max_new_tokens=256 captures all but ~10 training examples
    full_ids = torch.cat([encoded_few_shot_prompt, input_ids], axis=1).to(device)
    generated_ids = model.generate(
        input_ids=full_ids.long(), 
        do_sample=True, 
        temperature=0.2, 
        max_new_tokens=256, 
        #pad_token_id=tokenizer.eos_token_id
    )

    # Isolate one program completion 
    start_idx = encoded_few_shot_prompt.size()[1]
    completion = tokenizer.decode(generated_ids[0, start_idx:])
    answer_locs = re.search('answer.*?\n', completion)
    if answer_locs: 
        program = completion[:answer_locs.span()[1]]
    else: 
        program = completion 
    answer = semisafe_evaluate(program, 'answer', 1)
    if answer is float: 
        if abs((answer - answer_sol) / answer) < 0.01: 
            num_correct += 1
        no_errors += 1

    # Writes results to a file
    to_dump += "\n" + "#"*20 + "\n"
    to_dump += "PROMPT: \n"
    to_dump += tokenizer.decode(full_ids.squeeze(), skip_special_tokens=True)
    to_dump += "\nGENERATED COMPLETION: \n" 
    to_dump += completion
    to_dump += "\nLABEL COMPLETION:\n"
    to_dump += tokenizer.decode(code_sol.squeeze(), skip_special_tokens=True)
    to_dump += "\nANSWER: " + str(answer) + "\n"
    to_dump += "\nLABEL ANSWER: " + str(answer_sol.item()) + "\n"

    break 


    
accuracy = num_correct / len(data) 
execution_rate = no_errors / len(data)

to_dump += "\n" + "ACCURACY: " + str(accuracy) 
to_dump += "\nEXECUTION_RATE: " + str(execution_rate) 

with open(output_file, "a") as fle: 
    fle.write(to_dump)

print(accuracy)
