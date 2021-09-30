import torch 
import json 
from pathlib import Path 

def read_mathqapython(path): 
    path = Path(path)
    with open(path, 'rb') as f: 
        mathqapython_list = json.load(f)

    texts = [instance['text'] for instance in mathqapython_list]
    codes = [instance['code'] for instance in mathqapython_list]
    answers = [instance['answer'] for instance in mathqapython_list]

    return texts, codes, answers

class MathQAPython(torch.utils.data.Dataset): 
    def __init__(self, text_encoding, code_encoding, answer): 
        self.text_encoding = text_encoding
        self.code_encoding = code_encoding 
        self.answer = answer 
