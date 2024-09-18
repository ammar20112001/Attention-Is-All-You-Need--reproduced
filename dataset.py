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
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'

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
        
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(ds_src_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.int64)
            # <sos> ...sentence tokens... <eos> <pad>...
        ], dim=0)

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(ds_tgt_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.int64)
            # <sos> ...sentence tokens... <pad>...
        ], dim=0)

        labels = torch.cat([
            torch.tensor(ds_tgt_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.int64)
            # <sos> ...sentence tokens... <pad>...
        ], dim=0)

        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() # (1, 1, seq_len)
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & BilingualDataset.causal_mask(decoder_input.size(0)) # (1, seq_len) & (1, seq_len, seq_len)

        return {
            'encoder_input': encoder_input,
            'dencoder_input': decoder_input,
            'labels': labels,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask
        }

        #return self.ds[idx], ds_src_tokens, ds_tgt_tokens
    
    @staticmethod
    def tokenize_text(sentence):
        return [0,1,2]
    
    @staticmethod
    def get_tokenizer():
        pass

    @staticmethod
    def causal_mask(size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0



def get_ds(config):
    ds = load_dataset(config['dataset'])
    return ds['train']


if __name__ == '__main__':
    ds = BilingualDataset(config)
    print(f'Length of dataset: {ds.__len__()}', ds.__getitem__(random.randrange(0, ds.__len__())), sep='\n')