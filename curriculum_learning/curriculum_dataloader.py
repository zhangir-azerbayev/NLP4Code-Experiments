import torch 
import json 
from pathlib import Path 

def read_mathqapython(path): 
    path = Path(path)
    with open(path, 'rb') as f: 
        mathqapython_list = json.load(f)


    return mathqapython_list

class MathQAPython(torch.utils.data.Dataset): 
    def __init__(self, instance_list, tokenizer, max_length, scorer, pacer): 
        # Data is sorted by the scoring function 
        self.data = scorer.sort(instance_list)
        self.tokenizer = tokenizer 
        self.max_length = max_length
        self.pacer = pacer 
    

    def __getitem__(self, idx): 
        instance = self.data[idx]
        text = instance['text'] 
        solution = instance['text'] + '\n' + instance['code'] 
        answer = instance['answer']

        text_encode = self.tokenizer(text, 
                max_length=self.max_length, truncation=True, 
                padding='max_length', return_tensors='pt')
        solution_encode = self.tokenizer(solution, 
                max_length=self.max_length, truncation=True, 
                padding='max_length', return_tensors='pt')
        text_ids = text_encode['input_ids'].squeeze()
        solution_ids = solution_encode['input_ids'].squeeze()
        solution_attn = solution_encode['attention_mask'].squeeze()

        return text_ids.long(), solution_ids.long(), solution_attn.long(), answer


    def __len__(self): 
        raise NotImplementedError # Should vary with the pacing function 

    def __iter__(self): 
        raise NotImplementedError 
