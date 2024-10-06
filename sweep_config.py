def sweep_configuration():
    return {
        
        # Sweep method
        'method': 'random',
        
        # Metric and goal
        'metric': {
            'name': 'LOSS_VAL',
            'goal': 'minimize'
        },

        'parameters': {
            
            # Dataset configurations
            "dataset": {'value': "kaitchup/opus-English-to-French"},
            "lang_src": {'value': "en"},
            "lang_tgt": {'value': "it"},
            "datasource": {'value': 'opus_books'},
            # Tokenizer
            'tokenizer': {'value': 'bert-base-uncased'},
            # Data loader configurations
            'batch_size': {'value': 64},
            # Data split configurations
            'train_frac': {'value': 0.9},
            'test_val_frac': {'value': 0.5},
            'train_rows': {'value': 10_000}, # Set value to `False` to laod full dataset
            'test_val_rows': {'value': 1_000},
            
            # Transformer model configurations
            'd_model': {'value': 256}, # In paper: 512
            'heads': {'value': 4}, # In paper: 8
            'n_stack': {'value': 3}, # In paper: 6
            'max_seq_len': {'value': 256}, # In paper: 512
            'src_vocab_size': {'value': 40_000},
            'tgt_vocab_size': {'value': 40_000},
            'dropout': {'value': 0.1}, # In paper: 0.1
            'd_fc': {'value': 1024}, # In paper: 2048
            'enc_max_seq_len': {'value': 256}, # In paper: 512
            'dec_max_seq_len': {'value': 256}, # In paper: 512

            # Training configurations
            # Learning rate
            'lr': {'value': 0.001},
            'optimizer': {'value': 'Adam'}, # Hard coded

            # W&B configs
            'log_text_len': {'value': 15}
        }
    }