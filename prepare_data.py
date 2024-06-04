# Download wikitext103 files from: https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-103-raw-v1
import pandas as pd
from utils import save_parquet_as_txt, load_txt, save_txt, save_embedded_data

val_source, val_dest = 'wikitext103/validation-00000-of-00001.parquet', 'wikitext103/validation.txt'
train01_source, train01_dest = 'wikitext103/train-00000-of-00002.parquet', 'wikitext103/train01.txt'
train02_source, train02_dest = 'wikitext103/train-00001-of-00002.parquet', 'wikitext103/train02.txt'
test_source, test_dest = 'wikitext103/test-00000-of-00001.parquet', 'wikitext103/test.txt'

save_parquet_as_txt(val_source, val_dest)
save_parquet_as_txt(train01_source, train01_dest)
save_parquet_as_txt(train02_source, train02_dest)
save_parquet_as_txt(test_source, test_dest)

train1 = load_txt(train01_dest) 
train2 = load_txt(train02_dest)

train_data = train1 + train2
train_dest = 'wikitext103/train_full.txt'

save_txt(train_data, train_dest)

save_embedded_data(train_dest, 'wikitext103/train_embd.bin')
save_embedded_data(val_dest, 'wikitext103/val_embd.bin')
save_embedded_data(test_dest, 'wikitext103/test_embd.bin')