import random
from config import config

import random

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from datasets import load_dataset


config = config()
tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
special_tokens_dict = {'bos_token': '<sos>', 'eos_token': '<eos>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
tokenizer.add_special_tokens(special_tokens_dict)


class BilingualDataset(Dataset):
    
    def __init__(self, config) -> None:
        super().__init__()
        # Initializing tokenizer
        #self.tokenizer = BilingualDataset.get_tokenizer(config)

        # Create dataset
        self.ds = BilingualDataset.get_ds(config, tokenize=True)
        self.enc_max_seq_len = config['enc_max_seq_len']
        self.dec_max_seq_len = config['dec_max_seq_len']

        # Initializing special tokens
        self.sos_token = torch.tensor(tokenizer('<sos>')['input_ids'][1:-1])
        self.eos_token = torch.tensor(tokenizer('<eos>')['input_ids'][1:-1])
        self.pad_token = torch.tensor(tokenizer('<pad>')['input_ids'][1:-1])
        self.split_char = tokenizer('###>')['input_ids'][1:-1]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        '''
        Arguments:
            idx: index of dataset row

        Operations:
            - Parses source and target sentences
            - Tokenize source and target sentences
            - Define number of padding tokens to be added to encoder and decoder inputs based on sequence length
            - Create the folloing items that will be returned:
                - encoder_input
                - decoder_input
                - decoder_labels
                - encoder_mask
                - decoder_mask

        Returns:
            {
                encoder_input
                decoder_input
                decoder_labels
                encoder_mask
                decoder_mask
                source_sentence
                target_sentence
            }
        '''

        # Parse source and target sentences
        sentence = self.ds['train'][idx]['input_ids']

        for i in range(len(sentence)):
            if sentence[i] == self.split_char[0]:
                ds_src_tokens = sentence[:i]
                ds_tgt_tokens = sentence[i+4:]
                break

        # Length of padding tokens in encoder and decoder inputs
        enc_num_pad_tokens = self.enc_max_seq_len - len(ds_src_tokens) - 2 # (-) <sos> and <eos>
        dec_num_pad_tokens = self.dec_max_seq_len - len(ds_tgt_tokens) - 1 # (-) <sos> in decoder input or <eos> in decoder labels

        if enc_num_pad_tokens < 0:
            raise ValueError(f"Sentence is too long. Expected {self.enc_max_seq_len}, received {len(ds_src_tokens)}")
        elif dec_num_pad_tokens < 0:
            raise ValueError(f"Sentence is too long. Expected {self.dec_max_seq_len}, received {len(ds_tgt_tokens)}")
        
        passs = True
        if passs:
            encoder_input = torch.cat([
                self.sos_token,
                torch.tensor(ds_src_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
                # <sos> ...sentence tokens... <eos> <pad>...
            ], dim=0)

            decoder_input = torch.cat([
                self.sos_token,
                torch.tensor(ds_tgt_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
                # <sos> ...sentence tokens... <pad>...
            ], dim=0)

            labels = torch.cat([
                torch.tensor(ds_tgt_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
                # <sos> ...sentence tokens... <pad>...
            ], dim=0)

            encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() # (1, 1, seq_len)
            decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & BilingualDataset.causal_mask(decoder_input.size(0)) # (1, seq_len) & (1, seq_len, seq_len)

            return {
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'labels': labels,
                'encoder_mask': encoder_mask,
                'decoder_mask': decoder_mask
            }

        return ds_src_tokens, ds_tgt_tokens
    

    @staticmethod
    def get_tokenizer(config):
        return AutoTokenizer.from_pretrained(config['tokenizer'])
    
    @staticmethod
    def tokenize_text(sentence):
        #x = (([0, 1, 2], [0, 1, 2]), [0, 1, 2], [0, 1, 2])
        return tokenizer(sentence['text'], add_special_tokens=False)#, tokenizer(sentence['text'][0].split(' ###>')[0])

    @staticmethod
    def causal_mask(size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0

    @staticmethod
    def get_ds(config, tokenize=True):
        ds = load_dataset(config['dataset'])
        if tokenize == True:
            ds = ds.map(BilingualDataset.tokenize_text, batched=True)      
        return ds


if __name__ == '__main__':
    ds = BilingualDataset(config)
    x = ds.__getitem__(random.randrange(0, ds.__len__()))
    for key, value in x.items():
        print(key, value)

    print('\n')
    print(tokenizer.decode(x['encoder_input']), tokenizer.decode(x['decoder_input'], sep='\n'))