import torch 
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import TrainingArguments, Trainer 

from dataloader import read_mathqapython, MathQAPython

print('loading data and configuring tokenizer')
data = read_mathqapython('data/mathqapython_train.json')

tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
tokenizer.pad_token = tokenizer.eos_token 

max_text_length = max([len(tokenizer.encode(datum['text'])) for datum in data])
max_code_length = max([len(tokenizer.encode(datum['code'])) for datum in data])
max_length = max([max_text_length, max_code_length])
print('max text_length: ', max_text_length, 'max_code_length: ', max_code_length)

train_set = MathQAPython(data, tokenizer, max_length, max_length)

print('loading model')
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

print('initializing training')

training_args = TrainingArguments(output_dir='./results',
                                  num_train_epochs=5,
                                  per_device_train_batch_size=16, 
                                  logging_steps=500,
                                  save_steps=1000,
                                  weight_decay=0.01,
                                  warmup_steps = 100, 
                                  logging_dir='./logs')

def data_collator(data):
    return {'input_ids': torch.stack([f[0] for f in data]),
            'labels': torch.stack([f[1] for f in data])
           }

Trainer(model=model, args=training_args, train_dataset=train_set, 
        data_collator=data_collator).train()
