import os
import json
import configparser

from data_builder import DataBuilder
from transformers import *
from tokenizers import *

class ProcessBertBackend:

    def __init__(self,
                 config_path,
                 input_folder_path,
                 training_folder_path,):

        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.input_folder_path = input_folder_path
        self.training_folder_path = training_folder_path
        self.vocab_size=int(self.config['TOKENIZER']['vocab_size'])
        self.max_length=int(self.config['TOKENIZER']['max_length'])
        self.truncate_longer_samples=bool(self.config['TOKENIZER']['truncate_longer_samples'])
        self.data_builder = DataBuilder(input_folder_path=self.input_folder_path,
                                        output_folder_path=self.training_folder_path)
        self.dataset, self.training_data_paths = self.data_builder.load_data()
        self.tokenizer = None
        self.test_dataset = None
        self.train_dataset = None

    def train_tokenizer(self,):
        model_path = self.config['TRAINING']['model_path']
        special_tokens = eval(self.config['TOKENIZER']['special_tokens'])
        files = self.training_data_paths
        tokenizer = BertWordPieceTokenizer()
        tokenizer.train(files=files, vocab_size=self.vocab_size, special_tokens=special_tokens)
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.save_model(model_path)
        with open(os.path.join(model_path, "config.json"), "w") as f:
            tokenizer_cfg = {
                "do_lower_case": True,
                "unk_token": "[UNK]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
                "model_max_length": self.max_length,
                "max_len": self.max_length,
            }
            json.dump(tokenizer_cfg, f)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        return

    def tokenize_dataset(self, ):

        d = self.dataset.train_test_split(test_size=0.1)
        # the encode function will depend on the truncate_longer_samples variable
        encode = self.__encode_with_truncation if self.truncate_longer_samples else self.__encode_without_truncation
        # tokenizing the train dataset
        train_dataset = d["train"].map(encode, batched=True)
        # tokenizing the testing dataset
        test_dataset = d["test"].map(encode, batched=True)
        if self.truncate_longer_samples:
            # remove other columns and set input_ids and attention_mask as
            train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
            test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        else:
            test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
            train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
            train_dataset = train_dataset.map(self.__group_texts, batched=True, batch_size=2_000,
                                              desc=f"Grouping texts in chunks of {self.max_length}")
            test_dataset = test_dataset.map(self.__group_texts, batched=True, batch_size=2_000,
                                            num_proc=4, desc=f"Grouping texts in chunks of {self.max_length}")


        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def load_model(self):

        model_config = BertConfig(vocab_size=self.vocab_size, max_position_embeddings=self.max_length)
        model = BertForMaskedLM(config=model_config)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.2
        )
        args = dict(self.config['TRAINING'])
        training_args = TrainingArguments(
            output_dir=args['model_path'],  # output directory to where save model checkpoint
            evaluation_strategy=args['evaluation_strategy'],  # evaluate each `logging_steps` steps
            overwrite_output_dir=bool(args['overwrite_output_dir']),
            num_train_epochs=int(args['num_train_epochs']),  # number of training epochs, feel free to tweak
            per_device_train_batch_size=int(args['per_device_train_batch_size']),  # the training batch size, put it as high as your GPU memory fits
            gradient_accumulation_steps=int(args['gradient_accumulation_steps']), #accumulating the gradients before updating the weights
            per_device_eval_batch_size=int(args['per_device_eval_batch_size']),  # evaluation batch size
            logging_steps=int(args['logging_steps']),  # evaluate, log and save model checkpoints every 1000 step
            save_steps=int(args['save_steps']),
            load_best_model_at_end=bool(args['load_best_model_at_end']),  # whether to load the best model (in terms of loss) at the end of training
            # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
        )

        print('begin to train the model')
        trainer.train()
        print('model training ended')

        model.save_pretrained(args['model_path'])

        return

    def __encode_with_truncation(self, examples):
        """Mapping function to tokenize the sentences passed with truncation"""
        return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.max_length, return_special_tokens_mask=True)

    def __encode_without_truncation(self, examples):
        """Mapping function to tokenize the sentences passed without truncation"""
        return self.tokenizer(examples["text"], return_special_tokens_mask=True)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def __group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.max_length:
            total_length = (total_length // self.max_length) * self.max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.max_length] for i in range(0, total_length, self.max_length)]
            for k, t in concatenated_examples.items()
        }
        return result






