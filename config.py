def configuration():
    return {
        "dataset": "kaitchup/opus-English-to-French",
        "lang_src": "en",
        "lang_tgt": "it",
        "datasource": 'opus_books',

        'batch_size': 32,

        'train_frac': 0.9,
        'test_val_frac': 0.5,
        'train_rows': 10_000,
        'test_val_rows': 1_000,

        'tokenizer': 'bert-base-uncased',
        
        'enc_max_seq_len': 256, # In paper: 512
        'dec_max_seq_len': 256, # In paper: 512

        # Transformer model architecture details
        'd_model': 256, # In paper: 512
        'heads': 4, # In paper: 8
        'n_stack': 3, # In paper: 6
        'max_seq_len': 256, # In paper: 512
        'src_vocab_size': 40_000,
        'tgt_vocab_size': 40_000,
        'dropout': 0.1, # In paper: 0.1
        'd_fc': 1024 # In paper: 2048
    }