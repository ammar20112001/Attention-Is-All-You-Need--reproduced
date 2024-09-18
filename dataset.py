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

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        '''
        Arguments:
            idx: index of dataset row

        Operations:
            - Parses source and targer sentences
            - Tokenize source and targer sentences
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
        ds_src_text_tokenized = tokenize_text(ds_src_text)
        ds_tgt_text_tokenized = tokenize_text(ds_tgt_text)

        return self.ds[idx], ds_src_text, ds_tgt_text


def get_ds(config):
    ds = load_dataset(config['dataset'])
    return ds['train']

def tokenize_text(sentence):
    return sentence


if __name__ == '__main__':
    ds = BilingualDataset(config)
    print(f'Length of dataset: {ds.__len__()}', ds.__getitem__(random.randrange(0, ds.__len__())), sep='\n')