import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

tokens = tokenizer.tokenize("hello world")
print(tokens)

ids = tokenizer.encode("hello world")
print(ids)
