import argparse
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm
import pickle
import wandb
import math
import types
import os
from model import GPT
from utils import load_embedded_data, generate_batches


def load_config(config_path):
    """Load the configuration from the given path."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--config", default="config.py", help="Path to the configuration file (cofig.py by default)")

    # Add arguments for each parameter
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--context_length", type=int, default=None, help="Context length")
    parser.add_argument("--n_layer", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size")
    parser.add_argument("--n_embd", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--bias", action="store_true", help="Use bias in the model")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=None, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--max_iters", type=int, default=None, help="Maximum iterations to train for")
    parser.add_argument("--lr_decay_iters", type=int, default=None, help="Decay learning rate upto this iteration")
    parser.add_argument("--warmup_iters", type=int, default=None, help="Warm up iterations")
    parser.add_argument("--eval_epochs", type=int, default=None, help="Number of evaluation epochs")
    parser.add_argument("--eval_intervel", type=int, default=None, help="Evaluation interval")
    parser.add_argument("--device", default=None, help="Device to use 'cuda' or 'cpu'")
    parser.add_argument("--save_chkpt_epoch", type=int, default=None, help="Save checkpoint every N epochs")
    parser.add_argument("--checkpoint_path", default=None, help="Path to save checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--load_checkpoint_path", default=None, help="Path to the checkpoint to resume training from")

    args = parser.parse_args()
    config = load_config(args.config)

    final_config = config.config
    for arg, value in vars(args).items():
        if value is not None and arg != "config":
            setattr(final_config, arg, value)

    return final_config


@torch.no_grad()
def get_loss(train_enc, val_enc, eval_epochs):
    model.eval()
    train_losses = torch.zeros(eval_epochs)
    val_losses = torch.zeros(eval_epochs)

    for iter in range(eval_epochs):
        if train_enc is not None:
            trainX, trainY = generate_batches(train_enc, config)
            _, train_loss = model(trainX, trainY)
            train_losses[iter] = train_loss.item()

        if val_enc is not None:
            valX, valY = generate_batches(val_enc, config)
            _, val_loss = model(valX, valY)
            val_losses[iter] = val_loss.item()

    train_loss_mean = train_losses.mean() if train_enc is not None else None
    val_loss_mean = val_losses.mean() if val_enc is not None else None
    model.train()
    return train_loss_mean, val_loss_mean


def checkpoint_model(model, config, train_loss, val_loss, epoch, path):
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    }, path)
    print(f"Model saved to {path}")


# Shamelessly copied from Karpathy's NanoGPT (https://github.com/karpathy/nanoGPT)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.lr * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.max_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.lr - config.min_lr)


def train(epochs, resume, save_checkpoint):
    for epoch in tqdm(range(epochs)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(it=epoch)
                
        for sub_epoch in range(config.grad_accum_steps):
            X, Y = generate_batches(train_enc, config)
            logits, loss = model(X, Y)
            latest_loss = loss.item()
            loss = loss / config.grad_accum_steps
            loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        actual_epoch = epoch + checkpoint['epoch'] if resume else epoch

        if epoch % config.eval_intervel == 0 or epoch % config.save_chkpt_epoch == 0 or epoch == epochs - 1:
            _, val_loss_mean = get_loss(train_enc, val_enc, config.eval_epochs)

        if epoch % config.save_chkpt_epoch == 0 and epoch != 0 and save_checkpoint:
            checkpoint_model(model, config, latest_loss, val_loss_mean, actual_epoch, 
                             os.path.join(config.checkpoint_path, f"ckpt_v4_ep{actual_epoch}.pt"))

        if epoch % config.eval_intervel == 0:
            print(f"Epoch {actual_epoch}, train loss: {latest_loss:.4f}, val loss: {val_loss_mean:.4f}")
        
        if wandb_logging:
                wandb.log({"Epoch": actual_epoch,
                        "train_loss":latest_loss,
                        "val_loss": val_loss_mean,
                        "lr": optimizer.param_groups[0]['lr']})

    if save_checkpoint:
        checkpoint_model(model, config, latest_loss, val_loss_mean, actual_epoch, 
                         os.path.join(config.checkpoint_path, f"ckpt_v4_ep{actual_epoch}.pt"))

    if wandb_logging: wandb.finish()
    return latest_loss, val_loss_mean


config = parse_args()

print("Using config:")
for attr, value in vars(config).items():
    print(f"    {attr}: {value}")

print("\nLoading data...\n")

train_enc = load_embedded_data("wikitext103/train_embd.bin")
val_enc = load_embedded_data("wikitext103/val_embd.bin")

train_enc = torch.tensor(train_enc)
val_enc = torch.tensor(val_enc)

model = GPT(config).to(config.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

# os.environ["WANDB_API_KEY"] = "key"

wandb_logging = True
wandb_project = 'test'
wandb_run_name = 'test02'

if wandb_logging:
    wandb.init(project=wandb_project, name=wandb_run_name,config=config)

if config.resume:
    assert config.load_checkpoint_path is not None, "Resume is True but load_checkpoint_path is None"
    checkpoint = torch.load(config.load_checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"\nResuming training from {config.load_checkpoint_path}\n")

if config.save_chkpt_epoch is not None:
    save_checkpoint = True
else:
    save_checkpoint = False

final_train_loss, final_val_loss = train(config.max_iters, resume=config.resume, save_checkpoint=save_checkpoint)
