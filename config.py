def configuration():
    return {
        "dataset": "kaitchup/opus-English-to-French",
        "lang_src": "en",
        "lang_tgt": "it",
        "datasource": 'opus_books',

        'batch_size': 32,

        'train_frac': 0.9,
        'test_val_frac': 0.5,

        'tokenizer': 'bert-base-uncased',
        
        'enc_max_seq_len': 512,
        'dec_max_seq_len': 512,

        # Transformer model architecture details
        'd_model': 512,
        'heads': 8,
        'n_stack': 6,
        'max_seq_len': 512,
        'src_vocab_size': 40_000,
        'tgt_vocab_size': 40_000,
        'dropout': 0.1,
        'd_fc': 2048
    }