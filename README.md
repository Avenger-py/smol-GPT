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

## Inference
## Evaluation
