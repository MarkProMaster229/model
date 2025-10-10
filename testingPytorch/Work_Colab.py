import torch
import torch.nn as nn
import torch.optim as optim
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import json
from huggingface_hub import hf_hub_download

from google.colab import drive
drive.mount('/content/drive')

CHECKPOINT_DIR = '/content/drive/MyDrive/transformer_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------------
# Настройки
# -----------------------------
MAX_LEN = 512
BATCH_SIZE = 10
CHUNK_SIZE = 500
EPOCHS = 70
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_file = hf_hub_download(
    repo_id="MarkProMaster229/Testing_model",
    filename="model_checkpoint.pt"
)


# -----------------------------
# Чанк-токенизация
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
with open("/content/dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
vocab_size = tokenizer.vocab_size
embedding_dim = 768


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
#на устройство
# -----------------------------
embedding_layer = embedding_layer.to(DEVICE)
transformer_encoder = transformer_encoder.to(DEVICE)
pos_encoding = pos_encoding.to(DEVICE)
output_layer = output_layer.to(DEVICE)

# Оптимизатор
optimizer = torch.optim.Adam(
    list(embedding_layer.parameters()) +
    list(transformer_encoder.parameters()) +
    list(pos_encoding.parameters()) +
    list(output_layer.parameters()),
    lr=1e-4
)

# -----------------------------
# Загружаем чекпоинт
# -----------------------------
start_epoch = 0
checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
embedding_layer.load_state_dict(checkpoint['embedding_state'])
pos_encoding.load_state_dict(checkpoint['pos_encoding_state'])
transformer_encoder.load_state_dict(checkpoint['transformer_state'])
output_layer.load_state_dict(checkpoint['output_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
start_epoch = checkpoint['epoch'] + 1
print(f"Модель загружена, продолжаем с эпохи {start_epoch}")
# -----------------------------
# Обучение с отладкой
# -----------------------------

SAVE_EVERY = 20  # сохранение каждые 20 эпох

for epoch in range(start_epoch, start_epoch + EPOCHS):
    running_loss = 0.0
    print(f"\n=== Эпоха {epoch + 1}/{start_epoch + EPOCHS} ===")

    for chunk_idx, (input_ids_chunk, attention_mask_chunk, target_ids_chunk) in enumerate(
            chunked_tokenizer(data, tokenizer, max_len=MAX_LEN, chunk_size=CHUNK_SIZE)
    ):
        dataset = TensorDataset(input_ids_chunk, attention_mask_chunk, target_ids_chunk)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for batch_idx, batch in enumerate(dataloader):
            batch_input_ids, batch_attention_mask, batch_target_ids = [x.to(DEVICE) for x in batch]
            padding_mask = (batch_attention_mask == 0)

            optimizer.zero_grad()

            # -----------------------------
            # Embedding
            # -----------------------------
            embedded = embedding_layer(batch_input_ids)
            print(f"[DEBUG] embedded shape: {embedded.shape}")  # [B, L, D] ?

            # Positional encoding
            embedded = embedded.transpose(0, 1)  # [L, B, D]
            embedded = pos_encoding(embedded)
            print(f"[DEBUG] embedded + pos_encoding shape: {embedded.shape}")  # [L, B, D]

            # Transformer
            transformer_output = transformer_encoder(embedded, src_key_padding_mask=padding_mask)
            transformer_output = transformer_output.transpose(0, 1)  # [B, L, D]
            print(f"[DEBUG] transformer_output shape: {transformer_output.shape}")

            # Вычисляем примерный расход памяти
            approx_memory_mb = transformer_output.element_size() * transformer_output.nelement() / 1024**2
            print(f"[DEBUG] Output memory approx: {approx_memory_mb:.2f} MB")

            # Linear layer
            logits = output_layer(transformer_output)
            print(f"[DEBUG] logits shape: {logits.shape}")

            # Loss
            loss = criterion(logits.view(-1, vocab_size), batch_target_ids.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_input_ids.size(0)
            print(f"[DEBUG] batch {batch_idx+1} loss: {loss.item():.6f}")

            # -----------------------------
            # Примеры текста
            # -----------------------------
            sample_input = batch_input_ids[0].cpu().tolist()
            sample_target = batch_target_ids[0].cpu().tolist()
            sample_pred = torch.argmax(logits[0], dim=-1).cpu().tolist()

            print(f"[DEBUG] Sample input:  {tokenizer.decode(sample_input, skip_special_tokens=True)[:100]}...")
            print(f"[DEBUG] Sample target: {tokenizer.decode(sample_target, skip_special_tokens=True)[:100]}...")
            print(f"[DEBUG] Sample pred:   {tokenizer.decode(sample_pred, skip_special_tokens=True)[:100]}...")

            del batch_input_ids, batch_attention_mask, batch_target_ids, embedded, transformer_output, logits
            torch.cuda.empty_cache()

    avg_loss = running_loss / len(data)
    print(f"\n=== Эпоха {epoch + 1} завершена — Avg Loss: {avg_loss:.6f} ===\n")

    # -----------------------------
    # Сохраняем каждые SAVE_EVERY эпох
    # -----------------------------
    if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == (start_epoch + EPOCHS):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save({
            'embedding_state': embedding_layer.state_dict(),
            'pos_encoding_state': pos_encoding.state_dict(),
            'transformer_state': transformer_encoder.state_dict(),
            'output_state': output_layer.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }, checkpoint_path)
        print(f"[DEBUG] Чекпоинт сохранён: {checkpoint_path}")

    # -----------------------------
    # Сохраняем каждые SAVE_EVERY эпох
    # -----------------------------
    if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == (start_epoch + EPOCHS):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save({
            'embedding_state': embedding_layer.state_dict(),
            'pos_encoding_state': pos_encoding.state_dict(),
            'transformer_state': transformer_encoder.state_dict(),
            'output_state': output_layer.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }, checkpoint_path)
        print(f"Чекпоинт сохранён: {checkpoint_path}")