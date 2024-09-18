import random
from config import config

import random

import torch
from torch.utils.data import Dataset, DataLoader


from datasets import load_dataset


config = config()

class BilingualDataset(Dataset):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.ds = get_ds(config)
        self.enc_max_seq_len = config['enc_max_seq_len']
        self.dec_max_seq_len = config['dec_max_seq_len']

        # Initializing tokenizer
        tokenizer = config['tokenizer']

        # Initializing special tokens
        self.sos_token = None
        self.eos_token = None
        self.tgt_token = None

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
        ds_src_text = self.ds[idx]['text'].split(' ###>')[0]
        ds_tgt_text = self.ds[idx]['text'].split(' ###>')[1]

        # Tokenize sentence
        ds_src_tokens = BilingualDataset.tokenize_text(ds_src_text)
        ds_tgt_tokens = BilingualDataset.tokenize_text(ds_tgt_text)

        # Length of padding tokens in encoder and decoder inputs
        enc_num_pad_tokens = self.enc_max_seq_len - len(ds_src_tokens) - 2 # (-) <sos> and <eos>
        dec_num_pad_tokens = self.dec_max_seq_len - len(ds_tgt_tokens) - 1 # (-) <sos> in decoder input or <eos> in decoder labels

        if enc_num_pad_tokens < 0:
            raise ValueError(f"Sentence is too long. Expected {self.enc_max_seq_len}, received {len(ds_src_tokens)}")
        elif dec_num_pad_tokens < 0:
            raise ValueError(f"Sentence is too long. Expected {self.dec_max_seq_len}, received {len(ds_tgt_tokens)}")
        
        encoder_input = torch.cat[
            self.sos_token,
            torch.tensor(ds_src_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_tokens] * enc_num_pad_tokens, dtype=torch.int64)
        ] # <sos> ...sentence tokens... <eos> <pad>...

        return self.ds[idx], ds_src_tokens, ds_tgt_tokens
    
    @staticmethod
    def tokenize_text(sentence):
        return sentence
    
    @staticmethod
    def get_tokenizer():
        pass


def get_ds(config):
    ds = load_dataset(config['dataset'])
    return ds['train']


if __name__ == '__main__':
    ds = BilingualDataset(config)
    print(f'Length of dataset: {ds.__len__()}', ds.__getitem__(random.randrange(0, ds.__len__())), sep='\n')