# {"<pad>":0, "<s>":1, "</s>":2, "<unk>":3}
special_tokens = ['<pad>', '<s>', '</s>', '<unk>']

PAD = 0
max_len = 128
base_hyper_params = {"N": 6,
                     "d_model": 512,
                     "d_ff": 2048,
                     "head": 8,
                     "d_k": 64,
                     "d_v": 64,
                     "dropout_rate": 0.1,
                     "label_smoothing": 0.1,
                     "train_steps": 100000}
big_hyper_params = {"N": 6,
                    "d_model": 1024,
                    "d_ff": 4096,
                    "head": 16,
                    "d_k": 64,
                    "d_v": 64,
                    "dropout_rate": 0.3,
                    "label_smoothing": 0.1,
                    "train_steps": 300000}
batch_size = 58
warmup_step = 4000
ende_vocab_size = 37000
enfr_vocab_size = 32000
device = None