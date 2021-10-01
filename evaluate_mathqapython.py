import sys

from transformers import RobertaTokenizer, T5ForConditionalGeneration
from dataloader import read_mathqapython, MathQAPython 

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

if sys.argv[1] == 'test': 
    data = read_mathqapython('data/mathqapython_test.json')
elif sys.argv[1] == 'dev': 
    data = read_mathqapython('data/mathqapython_dev.json')
else: 
    raise ValueError("Invalid test/dev argument")

test_set = MathQAPython(data, tokenizer, 256, 256)

instance = next(enumerate(test_set))[1]
print(instance)

ids = instance['text_ids']
mask = instance['text_mask']
generated_ids = model.generate(
        input_ids=ids, 
        attention_mask=mask, 
        max_length=512
        )

div = '#'*20 

print('text\n', tokenizer.decode(ids, skip_special_tokens=False))
print(div)
print('dataloader output: \n', instance)
print(div)
print('model output: \n', tokenizer.decode(generated_ids[0], skip_special_tokens=False))

