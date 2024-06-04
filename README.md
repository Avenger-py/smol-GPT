# smol-GPT
Built a small (22M params) GPT 2 like transformer language model from scratch and trained it on Wikitext103 dataset.

## Results
Trained on T4 gpu on Kaggle. `val_loss` can be brought down further by simply training for more iterations or by changing learning rate or other hyper-parameters in `config.py`.

GPT-22M model training run visualized:

![smol-GPT](assets/GPT-22M_training_run.png)

## Dependencies
```
pip install torch numpy tiktoken wandb tqdm sklearn pandas
```
## Dataset
I have used wikitest103 dataset with the following split:
- Train: 122M tokens
- Validation: 250K tokens
- Test: 290k tokens

Instructions to prepare the dataset:
- Go to [https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-103-raw-v1] and download all the 4 parquet files
- Place them in the `wikitext103` folder
- run `prepare_data.py`
- 3 files will be generated: `train_embd.bin`, `val_embd.bin`, `test_embd.bin`. Keep the files here only.

## Training
GPU is recommended for training. The code is flexible and you can increase/decrease the model parameters. Here's how to train the model:
```sh
python train.py --config path_to_config(config.py by default)
```
You can also set the following training params when executing `train.py` or provide them in your config file:

```
Training parameters

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the configuration file (cofig.py by default)
  --batch_size BATCH_SIZE
                        Batch size
  --grad_accum_steps GRAD_ACCUM_STEPS
                        Gradient accumulation steps
  --context_length CONTEXT_LENGTH
                        Context length
  --n_layer N_LAYER     Number of transformer layers
  --n_head N_HEAD       Number of attention heads
  --vocab_size VOCAB_SIZE
                        Vocabulary size
  --n_embd N_EMBD       Embedding dimension
  --bias                Use bias in the model
  --dropout DROPOUT     Dropout rate
  --lr LR               Learning rate
  --min_lr MIN_LR       Minimum learning rate
  --weight_decay WEIGHT_DECAY
                        Weight decay
  --max_iters MAX_ITERS
                        Maximum iterations to train for
  --lr_decay_iters LR_DECAY_ITERS
                        Decay learning rate upto this iteration
  --warmup_iters WARMUP_ITERS
                        Warm up iterations
  --eval_epochs EVAL_EPOCHS
                        Number of evaluation epochs
  --eval_intervel EVAL_INTERVEL
                        Evaluation interval
  --device DEVICE       Device to use 'cuda' or 'cpu'
  --save_chkpt_epoch SAVE_CHKPT_EPOCH
                        Save checkpoint every N epochs
  --checkpoint_path CHECKPOINT_PATH
                        Path to save checkpoints
  --resume              Resume training
  --load_checkpoint_path LOAD_CHECKPOINT_PATH
                        Path to the checkpoint to resume training from
```
## Inference
## Evaluation
