import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer

# 1. Токенизация
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
text = ["hello world frfr - furry cute милые лисята"]  # список, чтобы был батч - 1

encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
input_ids = encoded['input_ids']           # [batch, seq_len]
attention_mask = encoded['attention_mask'] # [batch, seq_len]

print("Input IDs:", input_ids)
print("Shape:", input_ids.shape)  # torch.Size([1, seq_len])

# 2. Слой эмбеддингов
vocab_size = tokenizer.vocab_size  # лучше брать прямо из токенизатора
embedding_dim = 768
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# 3. Получаем эмбеддинги батча
embedded = embedding_layer(input_ids)  # batch, seq_len, embedding_dim это сам тензер ес че епта
print("Embedded shape:", embedded.shape)
torch.save(embedded, "tensor.pt")#спасибо тому человеку!

print(embedded[0,0:20])