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

## Training

## Inference
## Evaluation
