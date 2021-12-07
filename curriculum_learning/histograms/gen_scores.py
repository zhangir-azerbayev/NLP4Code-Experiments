import sys
import os 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from score import LenScore, OperationScore, UniqueOperationScore
import re
import json

def read_inferences(path): 
    """Output is list of instances with keys {"code", "answer", "label_answer"}"""

    with open(path) as f: 
        lines = f.readlines() 
        pointer = 0 

        instance_list = []
        
        instance = dict() 
        print('starting while loop')
        i = 0 
        while i < len(lines): 
            if re.search("LABEL COMPLETION:", lines[i]): 
                i+=2
                program = ""
                while not re.search('^answer', lines[i]): 
                    program = program + lines[i]
                    i+=1
                program = program + lines[i]
                i+=1 

                instance["code"] = program 

            elif re.search('^ANSWER', lines[i]): 
                if re.search('(failed|not)', lines[i]): 
                    instance["answer"] = "failed"
                else: 
                    instance["answer"] = float(lines[i].split()[-1])
                i+=1 

            elif re.search('^LABEL ANSWER', lines[i]): 
                instance["label_answer"] = float(lines[i].split()[-1])
                instance_list.append(instance)
                i+=1 
                instance = dict()
            else: 
                i+=1

    return instance_list 

# Script 
instance_list = read_inferences("../../results/125M_finetuned/inferences.txt")

lenscore = LenScore() 
opscore = OperationScore()
uopscore = UniqueOperationScore() 

scores = [{"pass_k": 0 if instance["answer"]=='failed' or abs((instance["label_answer"] - instance["answer"])/instance["label_answer"]) > 0.01 else 1, 
           "len_score": lenscore.score(instance), 
           "operation_score": opscore.score(instance), 
           "unique_operation_score": uopscore.score(instance)}
           for instance in instance_list]

print(scores)

with open('scores.json', "w") as f: 
    json.dump(scores, f)

                
