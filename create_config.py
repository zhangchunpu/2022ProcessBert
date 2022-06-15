import configparser

config = configparser.ConfigParser()

config['TOKENIZER'] = {
    'max_length': 128,
    'truncate_longer_samples': True,
    'vocab_size': 30522,
    'special_tokens': ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
}
config['TRAINING'] = {
    'model_path':'models',
    'evaluation_strategy':'steps',
    'overwrite_output_dir':True,
    'num_train_epochs': 100,
    'per_device_train_batch_size': 256,
    'gradient_accumulation_steps': 10,
    'per_device_eval_batch_size': 256,
    'logging_steps': 1000,
    'save_steps': 20000,
    'load_best_model_at_end': True,
    'save_total_limit': 3,
}
with open('config.ini', 'w') as file:
    config.write(file)