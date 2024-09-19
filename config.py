def config():
    return {
        'dataset': 'kaitchup/opus-Swedish-to-English',
        "lang_src": "en",
        "lang_tgt": "it",
        "datasource": 'opus_books',

        'tokenizer': 'bert-base-uncased',
        
        'enc_max_seq_len': 512,
        'dec_max_seq_len': 512,
    }