import tiktoken
import pandas as pd
import pickle
import torch

def load_txt(path):
    with open(path, 'r',  encoding='utf-8') as f:
        var = f.read()
    print(f"Loaded {path}")
    return var


def save_parquet_as_txt(source, dest):
    df = pd.read_parquet(source)
    df.to_csv(dest, index=False)
    print(f"Converted {source} to {dest}")


def save_txt(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)
    print(f"Saved {path}")


def encode_data(data):
    encd = tiktoken.get_encoding("gpt2")
    data_enc = encd.encode_ordinary(data)
    print(f"Encoded")
    return data_enc


def save_embedded_data(source, dest):
    data = load_txt(source)
    data_enc = encode_data(data)
    with open(dest, "wb") as file:
        pickle.dump(data_enc, file)
    print(f"Saved {dest}")


def load_embedded_data(path):
    with open(path, "rb") as file:
        loaded_list = pickle.load(file)
    print(f"Loaded {path}")
    return loaded_list


def generate_batches(data, config):
    idx = torch.randint(len(data) - config.context_length, (config.batch_size,))
    x = torch.stack([data[i : i + config.context_length] for i in idx])
    y = torch.stack([data[i + 1 : i + config.context_length + 1] for i in idx])
    x, y = x.to(config.device), y.to(config.device)
    return (x, y)


def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_model_device(model):
    return next(model.parameters()).is_cuda