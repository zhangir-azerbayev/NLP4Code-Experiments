import sys 
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch 
from torch.utils.tensorboard import SummaryWriter 

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import TrainingArguments, Trainer 
from transformers.integrations import TensorBoardCallback

from dataloader import read_mathqapython, MathQAPython

name = sys.argv[1]

print('loading data and configuring tokenizer')
data = read_mathqapython('data/mathqapython_train.json')

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token 
max_length = 450

train_set = MathQAPython(data, tokenizer, max_length)

print('loading model')
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

print('initializing training')

training_args = TrainingArguments(output_dir="./train_results/{}".format(name),
                                  num_train_epochs=50,
                                  per_device_train_batch_size=16, 
                                  logging_strategy="epoch",
                                  save_strategy="epoch",
                                  weight_decay=0.01,
                                  warmup_steps = 100, 
                                  )

def data_collator(data):
    return {'input_ids': torch.stack([f[1] for f in data]),
            'attention_mask': torch.stack([f[2] for f in data]), 
            'labels': torch.stack([f[1] for f in data])
           }

tb_writer = SummaryWriter(log_dir="./train_results/{}/tb_log".format(name))
tb_callback = TensorBoardCallback(tb_writer)

Trainer(model=model, args=training_args, train_dataset=train_set, 
        data_collator=data_collator, callbacks=[tb_callback]).train()
