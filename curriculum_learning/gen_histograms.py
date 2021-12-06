import matplotlib.pyplot as plt 
import json 

with open('scores.json') as f: 
    score_list = json.load(f) 


len_scores = [instance["len_score"] for instance in score_list]

plt.hist(len_scores)
plt.savefig('len_scores.pdf') 
