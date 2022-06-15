import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from transformers import *
from tokenizers import *

from processbert_backend import ProcessBertBackend

input_path = "/mnt/d/2022_zhangchunpu/ChemECorpusLarge/xml"
output_path = "/mnt/c/Users/hsluser/Desktop/2022_zhangchunpu/ChemECorpusLarge"
config_path = "config.ini"

print(torch.cuda.is_available())
print('current device:', torch.cuda.current_device())
print('device number:', torch.cuda.device_count())
# print('memory management:', torch.cuda.max_memory_allocated(device=0))

system = ProcessBertBackend(config_path=config_path, input_folder_path=input_path, training_folder_path=output_path)
system.train_tokenizer()
system.tokenize_dataset()
system.load_model()



