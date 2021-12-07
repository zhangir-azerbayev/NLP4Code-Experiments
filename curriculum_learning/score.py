from curriculum_dataloader import *
from transformers import GPT2Tokenizer
import re

class Score(): 
    def __init__(): 
        raise NotImplementedError 


    def score(self, instance): 
        """Return the difficulty score of a MathQAPython instance"""
        raise NotImplementedError 

    def sort(self, instance_list): 
        """Sort a list of instances ascending order of difficulty score"""
        raise NotImplementedError 



# Implement a bunch of scoring functions here

# Score = # of tokens in the solution
class LenScore(Score): 
    def __init__(self): 
        pass 

    def score(self, instance): 
        return len(instance["code"].split())
    
    def sort(self, instance_list):
        return sorted(instance_list, key=self.score)

class OperationScore(Score): 
    def __init__(self): 
        pass 

    def score(self, instance): 
        soln = instance["code"]
        # handle exceptional cases 
        soln = re.sub(r'\*\*',' \*\* ', soln)
        soln = re.sub(r',', ' ', soln)
        soln = re.sub(r'_',' ', soln)
        soln = re.sub(r'#',' ', soln)
        soln = re.sub(r'e-','', soln)

        # now tokenize
        tokens = re.split(r'\s|\(|\)', soln)
        # filter out empty strings
        tokens = [token for token in tokens if token != '']
        # filter out floats/ints
        tokens = [token for token in tokens if not token.replace(".", "", 1).isdecimal()]
        # filter out var names
        tokens = [token for token in tokens if not token.isalnum() or token == "max" or token == "min"]
        
        return len(tokens)
    
    def sort(self, instance_list):
        return sorted(instance_list, key=self.score)
        
# Score = # of unique operations in solution
class UniqueOperationScore(Score): 
    def __init__(self): 
        pass 

    def score(self, instance): 
        soln = instance["code"]
        # handle exceptional cases 
        soln = re.sub(r'\*\*',' \*\* ', soln)
        soln = re.sub(r',', ' ', soln)
        soln = re.sub(r'_',' ', soln)
        soln = re.sub(r'#',' ', soln)
        soln = re.sub(r'e-','', soln)

        # now tokenize
        tokens = re.split(r'\s|\(|\)', soln)
        # filter out empty strings
        tokens = [token for token in tokens if token != '']
        # filter out floats/ints
        tokens = [token for token in tokens if not token.replace(".", "", 1).isdecimal()]
        # filter out var names
        tokens = [token for token in tokens if not token.isalnum() or token == "max" or token == "min"]
        
        # filter out duplicate tokens
        tokens = list(set(tokens))

        # print(len(tokens), tokens)
        return len(tokens)
    
    def sort(self, instance_list):
        return sorted(instance_list, key=self.score)


class BetterLenScore(Score):
    def __init__(self):
        pass

    def score(self, instance):
        # tokenize differently, in the same way as the operation score
        soln = instance["code"]
        # handle exceptional cases 
        soln = re.sub(r'\*\*',' \*\* ', soln)
        soln = re.sub(r',', ' ', soln)
        soln = re.sub(r'_',' ', soln)
        soln = re.sub(r'#',' ', soln)
        soln = re.sub(r'e-','', soln)

        # now tokenize
        tokens = re.split(r'\s|\(|\)', soln)

        # filter out extraneous empty strings
        tokens = [token for token in tokens if token != '']

        return len(tokens)
    
    def sort(self, instance_list):
        return sorted(instance_list, key=self.score)

