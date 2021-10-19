import sys
import re

import torch
from torch.utils.data import DataLoader

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from dataloader import read_mathqapython, MathQAPython 

# Take a look at some data
data = read_mathqapython('data/mathqapython_dev.json')

data[35]

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token

max_text_length = max([len(tokenizer.encode(datum['text'])) for datum in data])
max_code_length = max([len(tokenizer.encode(datum['code'])) for datum in data])
print(max_text_length, max_code_length)

few_shot_prompt = "\n\n".join([example['text'] + '\n' + example['code'] for example in data[0:12] ]) + '\n\n'
print(len(tokenizer(few_shot_prompt)['input_ids']))
print(few_shot_prompt)

tokenizer(few_shot_prompt)['attention_mask']

test_set = MathQAPython(data, tokenizer, 256, 256)

loader = DataLoader(test_set, batch_size=1, shuffle=True)

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

num_correct = 0 
for batch in loader: 
    ids, gt, gt_answer = batch
    encoded_few_shot_prompt = tokenizer(few_shot_prompt, return_tensors="pt")['input_ids']
    few_shot_ids = torch.cat([encoded_few_shot_prompt, ids], axis=1)
    generated_ids = model.generate(
        input_ids=few_shot_ids, 
        do_sample=True,
        temperature=0.4, 
        max_length=2048
        )
    

    start_idx = encoded_few_shot_prompt.size()[1]
    completion = tokenizer.decode(generated_ids[0, start_idx:])
    program = completion[:re.search('answer.*?\n', completion).span()[1]]
    print(program)
    loc={}
    a = exec(program, globals(), loc)
    if abs((loc['answer']-gt_answer)/gt_answer) < 0.01: 
        num_correct += 1
        
print(num_correct/length(test_set))
    
    