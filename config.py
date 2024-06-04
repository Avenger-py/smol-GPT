import types

config = dict(
    batch_size = 32,
    grad_accum_steps = 8,
    context_length = 196, 
    N_layer = 6,
    N_head = 6,
    vocab_size = 50257,
    N_embd = 192,
    bias = False,
    dropout = 0.3,
    lr = 0.005,
    min_lr = 0.00001,
    weight_decay = 1e-2,
    max_iters = 50000, # total iters (total unique batches) ~ 1,916,994 (total tokens - ctx length)/batch_size
    lr_decay_iters = 50000,
    warmup_iters = 750,
    eval_epochs = 120,
    eval_intervel = 500,
    device = 'cuda',
    save_chkpt_epoch = 5000,
    checkpoint_path = './checkpoints',
    resume = False,
    load_checkpoint_path = None
)

config = types.SimpleNamespace(**config)