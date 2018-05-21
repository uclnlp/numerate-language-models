
def get_model_config(which):
    config = None
    if which == 'a1':
        # flat categ (bias)
        config = {
            'which': which,
            'emb_dim': 50,
            'numcoders': [2, 2],
            'output_mode': ['categorical'],  # <===
            'use_loc_embs': False,  # <===
            'use_hierarchical': False,  # <===
        }
    elif which == 'a2':
        # flat categ (bias + rnn)
        config = {
            'which': which,
            'emb_dim': 50,
            'numcoders': [2, 2],
            'output_mode': ['categorical'],
            'use_loc_embs': True,    # <===
            'use_hierarchical': False,    # <===
        }
    elif which == 'a3':
        # categ (bias)
        config = {
            'which': which,
            'emb_dim': 50,
            'numcoders': [2, 2],
            'output_mode': ['categorical'],
            'use_loc_embs': False,  # <===
            'use_hierarchical': True,    # <===
        }
    elif which == 'a4':
        # categ (bias + rnn)
        config = {
            'which': which,
            'emb_dim': 50,
            'numcoders': [2, 2],
            'output_mode': ['categorical'],
            'use_loc_embs': True,  # <===
            'use_hierarchical': True,    # <===
        }
    elif which == 'b1':
        # rnn
        config = {
            'which': which,
            'emb_dim': 50,
            'numcoders': [2, 2],
            'output_mode': ['rnn'],  # <===
            'use_loc_embs': False,
            'use_hierarchical': True,
        }
    elif which == 'b2':
        # cont (bias)
        config = {
            'which': which,
            'emb_dim': 50,
            'numcoders': [2, 2],
            'output_mode': ['continuous'],  # <===
            'use_loc_embs': False,
            'use_hierarchical': True,
        }
    elif which == 'c1':
        config = {
            'which': which,
            'emb_dim': 50,
            'numcoders': [2, 2],
            'output_mode': ['categorical', 'rnn'],  # <=== combo
            'use_loc_embs': True,
            'use_hierarchical': True,
        }
    elif which == 'c2':
        config = {
            'which': which,
            'emb_dim': 50,
            'numcoders': [2, 2],
            'output_mode': ['categorical', 'continuous'],  # <=== combo
            'use_loc_embs': True,
            'use_hierarchical': True,
        }
    elif which == 'c3':
        config = {
            'which': which,
            'emb_dim': 50,
            'numcoders': [2, 2],
            'output_mode': ['rnn', 'continuous'],  # <=== combo
            'use_loc_embs': True,
            'use_hierarchical': True,
        }
    elif which == 'c4':
        config = {
            'which': which,
            'emb_dim': 50,
            'numcoders': [2, 2],
            'output_mode': ['categorical', 'rnn', 'continuous'],  # <=== combo
            'use_loc_embs': True,
            'use_hierarchical': True,
        }
    return config