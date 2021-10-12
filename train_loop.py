import torch 
from transformers import GPTNeoForCausualLM, GPT2Tokenizer
from transformers import TrainingArguments, Trainer 

from dataloader import read_mathqapython, MathQAPython


data = read_mathqapython('data/mathqapython_train.json')

tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
tokenizer.pad_token = tokenizer.eos_token 

train_set = MathQAPython(data, tokenizer, 1792, 256)


model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

training_args = TrainingArguments(output_dir='./results',
                                  num_train_epochs=5,
                                  logging_steps=500,
                                  save_steps=5000,
                                  per_device_train_batch_size=2,
                                  weight_decay=0.01,
                                  warmup_steps = 100, 
                                  logging_dir='./logs')

def data_collator(data):
    return {'input_ids': torch.stack([f[0] for f in data]),
            'labels': torch.stack([f[1] for f in data])
           }

Trainer(model=model, args=training_args, train_dataset=train_set, 
        data_collator=data_collator).train()
