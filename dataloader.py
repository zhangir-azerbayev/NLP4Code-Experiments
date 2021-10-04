import torch 
import json 
from pathlib import Path 

def read_mathqapython(path): 
    path = Path(path)
    with open(path, 'rb') as f: 
        mathqapython_list = json.load(f)


    return mathqapython_list

class MathQAPython(torch.utils.data.Dataset): 
    def __init__(self, instance_list, tokenizer, text_len, code_len): 
        self.data = instance_list 
        self.tokenizer = tokenizer 
        self.text_len = text_len
        self.code_len = code_len
    

    def __getitem__(self, idx): 
        instance = self.data[idx]
        text = instance['text']
        code = instance['code']
        answer = instance['answer']

        text_encode = self.tokenizer.batch_encode_plus(text, max_length = self.text_len, pad_to_max_length=True, return_tensors='pt')
        code_encode = self.tokenizer.batch_encode_plus(code, max_length = self.code_len, pad_to_max_length=True, return_tensors='pt')

        text_ids = text_encode['input_ids'].squeeze()
        text_mask = text_encode['attention_mask'].squeeze()
        code_ids = code_encode['input_ids'].squeeze()
        code_mask = code_encode['attention_mask'].squeeze()

        return {
                'text_ids': text_ids.to(dtype=torch.long), 
                'text_mask': text_mask.to(dtype=torch.long), 
                'code_ids': code_ids.to(dtype=torch.long), 
                'code_mask': code_mask.to(dtype=torch.long), 
                'answer': answer
                }


    def __len__(self): 
        return len(self.data)
