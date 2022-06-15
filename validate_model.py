from transformers import *

tokenizer_model_path = 'models'
processbert_model_path = 'models'

# load the model checkpoint
model = BertForMaskedLM.from_pretrained(processbert_model_path)
# load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_model_path)

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
examples = [
  "The manufacturing of pharmaceutical [MASK] invariably involves solid-state processing stages, such as crystallization, transportation, storage, milling, mixing, granulation, and compression.",
  "The [MASK] was cloudy yesterday, but today it's rainy.",
  "The authors review selection parameters for optimal [MASK] of such materials, and discuss both the capabilities and limitations of mechanically-activated systems.I hope you enjoy reading these articles.Sincerely,Hamid GhandehariUniversity of UtahSalt Lake City, Utah, USA",
  "Recently there has been an explosion of new research designing biomaterials for intervening in immunological processes to achieve therapeutic [MASK] in applications ranging from chronic inflammation to autoimmunity to cancer."
]

examples_2 = ['my [MASK] is cute', 'he is good at [MASK]']
for example in examples:
  for prediction in fill_mask(example):
    print(f"{prediction['sequence']}, confidence: {prediction['score']}")
  print("="*50)