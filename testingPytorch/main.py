import torch
import torch.nn as nn
import torch.optim as optim
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import json

# -----------------------------
# Настройки
# -----------------------------
MAX_LEN = 512
BATCH_SIZE = 5
CHUNK_SIZE = 400
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "model_checkpoint.pt"

# -----------------------------
# Чанк-токенизация для экономии памяти
# -----------------------------
def chunked_tokenizer(data, tokenizer, max_len=512, chunk_size=100):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        texts = [item["input"] for item in chunk]
        targets = [item["target"] for item in chunk]

        encoded_input = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        encoded_target = tokenizer(
            targets,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        yield encoded_input['input_ids'], encoded_input['attention_mask'], encoded_target['input_ids']

# -----------------------------
# Загружаем данные
# -----------------------------
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
vocab_size = tokenizer.vocab_size
embedding_dim = 768

# -----------------------------
# Слои модели
# -----------------------------
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, embedding_dim: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)
    def forward(self, x):
        seq_len, batch_size, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        pos_embed = self.pos_embedding(positions)
        pos_embed = pos_embed.expand(-1, batch_size, -1)
        return x + pos_embed

pos_encoding = LearnedPositionalEncoding(MAX_LEN, embedding_dim)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Трансформер
nhead = 12
num_layers = 2
dim_feedforward = 2048
dropout = 0.1

transformer_layer = nn.TransformerEncoderLayer(
    d_model=embedding_dim,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    batch_first=False
)
transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
output_layer = nn.Linear(embedding_dim, vocab_size)

# Оптимизатор
optimizer = torch.optim.Adam(
    list(embedding_layer.parameters()) +
    list(transformer_encoder.parameters()) +
    list(pos_encoding.parameters()) +
    list(output_layer.parameters()),
    lr=1e-4
)

# -----------------------------
# Функция декодирования токенов
# -----------------------------
def decode_tokens(tokens):
    text = ""
    for t in tokens:
        if t.startswith("##"):
            text += t[2:]
        else:
            text += " " + t
    return text.strip()

# -----------------------------
# Подготовка таргетов
# -----------------------------
targets = [item["target"] for item in data]
referense = tokenizer(
    targets,
    padding='max_length',
    truncation=True,
    max_length=MAX_LEN,
    return_tensors='pt'
)
target_ids = referense['input_ids']
torch.save(target_ids, "inputReferense_ids.pt")

# -----------------------------
# Загружаем чекпоинт
# -----------------------------
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    embedding_layer.load_state_dict(checkpoint['embedding_state'])
    pos_encoding.load_state_dict(checkpoint['pos_encoding_state'])
    transformer_encoder.load_state_dict(checkpoint['transformer_state'])
    output_layer.load_state_dict(checkpoint['output_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Модель загружена, продолжаем с эпохи {start_epoch}")
else:
    print("Чекпоинт не найден, начинаем обучение с нуля")

# -----------------------------
# Переводим на устройство
# -----------------------------
embedding_layer = embedding_layer.to(DEVICE)
transformer_encoder = transformer_encoder.to(DEVICE)
pos_encoding = pos_encoding.to(DEVICE)
output_layer = output_layer.to(DEVICE)
# -----------------------------
# Обучение с отладкой
# -----------------------------
for epoch in range(start_epoch, start_epoch + EPOCHS):
    running_loss = 0.0
    print(f"\n=== Эпоха {epoch + 1}/{start_epoch + EPOCHS} ===")

    for chunk_idx, (input_ids_chunk, attention_mask_chunk, target_ids_chunk) in enumerate(
            chunked_tokenizer(data, tokenizer, max_len=MAX_LEN, chunk_size=CHUNK_SIZE)
    ):
        print(f"\n--- Чанк {chunk_idx + 1} / {len(data) // CHUNK_SIZE + 1} ---")
        dataset = TensorDataset(input_ids_chunk, attention_mask_chunk, target_ids_chunk)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for batch_idx, batch in enumerate(dataloader):
            batch_input_ids, batch_attention_mask, batch_target_ids = [x.to(DEVICE) for x in batch]
            padding_mask = (batch_attention_mask == 0)

            optimizer.zero_grad()

            # Эмбеддинги
            embedded = embedding_layer(batch_input_ids)
            print(f"[DEBUG] embedded shape: {embedded.shape}")  # batch, seq_len, embed_dim

            # Позиционное кодирование
            embedded = embedded.transpose(0, 1)  # seq_len, batch, embed_dim
            embedded = pos_encoding(embedded)
            print(f"[DEBUG] embedded + pos_encoding shape: {embedded.shape}")

            # Трансформер
            transformer_output = transformer_encoder(embedded, src_key_padding_mask=padding_mask)
            transformer_output = transformer_output.transpose(0, 1)  # batch, seq_len, embed_dim
            print(f"[DEBUG] transformer_output shape: {transformer_output.shape}")

            # Память выхода трансформера (примерно)
            batch_size, seq_len, emb_dim = transformer_output.shape
            mem_MB = batch_size * seq_len * emb_dim * 4 / 1024 ** 2
            print(f"[DEBUG] Output memory approx: {mem_MB:.2f} MB")

            # Линейный слой
            logits = output_layer(transformer_output)
            print(f"[DEBUG] logits shape: {logits.shape}")

            # Потери
            loss = criterion(logits.view(-1, vocab_size), batch_target_ids.view(-1))
            print(f"[DEBUG] batch {batch_idx + 1} loss: {loss.item():.6f}")

            # Backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_input_ids.size(0)

            # Демонстрация предсказаний
            pred_tokens = torch.argmax(logits, dim=-1)
            sample_input = tokenizer.decode(batch_input_ids[0], skip_special_tokens=True)
            sample_pred = tokenizer.decode(pred_tokens[0], skip_special_tokens=True)
            sample_target = tokenizer.decode(batch_target_ids[0], skip_special_tokens=True)
            print(f"[DEBUG] Sample input:  {sample_input[:50]}...")
            print(f"[DEBUG] Sample target: {sample_target[:50]}...")
            print(f"[DEBUG] Sample pred:   {sample_pred[:50]}...")

            # Очистка памяти
            del batch_input_ids, batch_attention_mask, batch_target_ids, embedded, transformer_output, logits
            torch.cuda.empty_cache()

    avg_loss = running_loss / len(data)
    print(f"\n=== Эпоха {epoch + 1} завершена — Avg Loss: {avg_loss:.6f} ===\n")

# -----------------------------
# Сохраняем чекпоинт
# -----------------------------
torch.save({
    'embedding_state': embedding_layer.state_dict(),
    'pos_encoding_state': pos_encoding.state_dict(),
    'transformer_state': transformer_encoder.state_dict(),
    'output_state': output_layer.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'epoch': epoch
}, CHECKPOINT_PATH)

print("Обучение завершено, чекпоинт сохранён.")
