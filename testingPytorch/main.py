import torch
import torch.nn as nn
import torch.optim as optim
import os
from transformers import AutoTokenizer

# 1. Токенизация
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
text = [(
    "Мамочка ты моя хорошая, хорошая ты моя мама, мамуля ты моя хорошая! "
    "Эх ты, матюшка ты моя! Мамочка ты моя! Что ж ты натворила-то?! "
    "Холодной хочешь стать что ли?! Остывшие щи хочешь мне преподать что ли? "
    "Мам! Тюря, ты каша-малаша в лепесточки! Эх ты! Мать-то моя! "
    "Лесом пошли бы, полем пошли бы! Сели бы! Спокойно, спросил бы, "
    "покакать можно, ты бы сказала бы: иди и покакай под кустик-то! "
    "И я б покакал бы! Посрал бы там! Обосрал всё! "
    "Говно бы все вытер пальцем, вытер бы! Мамулечка, потом бы листочком бы вытер! "
    "Я бы весь листочком был бы умытый, был бы!!!"
)]# список, чтобы был батч - 1
#forward pass
encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
input_ids = encoded['input_ids']           # [batch, seq_len]
attention_mask = encoded['attention_mask'] # [batch, seq_len]

torch.save(input_ids, "input_ids.pt")
torch.save(attention_mask, "attention_mask.pt")


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

#трансформер
src = embedded.transpose(0, 1) #теперь seq_len затем batch затем em bedding_dim - это требование трнсформера

transformerLayer = nn.TransformerEncoderLayer(
    d_model=embedding_dim,
    nhead=12,#ахринеть как все просто!!!! это типо я щас 12 голов внимания имею!!!
    dim_feedforward=2048, # внутренняя размерность FFN
    dropout=0.1, #Dropout - отключает 10% нейронов случайно при обучении
    batch_first=False#местами уже изменен тут - src = embedded.transpose(0, 1)

)
transformer_encoderLayer = nn.TransformerEncoder(transformerLayer,num_layers=2)
paddingMask = (attention_mask == 0)

outputTransformer0 = transformer_encoderLayer(
    src,
    src_key_padding_mask = paddingMask

)
outputTransformer = outputTransformer0.transpose(0, 1)# [S, B, E]
print(outputTransformer.shape)

#Линейный слой
vocab_size = tokenizer.vocab_size
embedding_dim = outputTransformer.shape[2]

output_layer = nn.Linear(embedding_dim,vocab_size)

logitOutputLayer = output_layer(outputTransformer) # [batch, seq_len, vocab_size]

# Для вероятностей
#probs = torch.softmax(logitOutputLayer, dim=-1)# softmax по словарю


#просто сопоставить самый вероятный токен
def decode_tokens(tokens):
    text = ""
    for t in tokens:
        if t.startswith("##"):
            text += t[2:]  # убираем ##
        else:
            text += " " + t
    return text.strip()

#обратное распространение(forward pass)

referense = tokenizer(
    ["Ты живешь последний час! Хочешь, я на одной ноге постою как цапля, хочешь?"],
    padding='max_length', truncation=True, max_length=input_ids.shape[1], return_tensors='pt'
)
target_ids = referense['input_ids']

torch.save(target_ids, "inputReferense_ids.pt")



criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


loss = criterion(
    logitOutputLayer.view(-1, vocab_size),
    target_ids.view(-1)
)

optimizer = torch.optim.Adam(list(embedding_layer.parameters()) +
                             list(transformer_encoderLayer.parameters()) +
                             list(output_layer.parameters()), lr=1e-4)

checkpoint_path = "model_checkpoint.pt"

if os.path.exists(checkpoint_path):
    # Загружаем модель и оптимизатор
    checkpoint = torch.load(checkpoint_path)
    embedding_layer.load_state_dict(checkpoint['embedding_state'])
    transformer_encoderLayer.load_state_dict(checkpoint['transformer_state'])
    output_layer.load_state_dict(checkpoint['output_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f" Модель загружена, продолжаем с эпохи {start_epoch}")
else:
    start_epoch = 0
    print(" Чекпоинт не найден, начинаем обучение с нуля")


epochNum = 20
for epoch in range(epochNum):
    optimizer.zero_grad()
    epochmy = start_epoch + epoch
    embedded = embedding_layer(input_ids)
    src = embedded.transpose(0, 1)

    outputTransformer = transformer_encoderLayer(src, src_key_padding_mask=(attention_mask == 0))
    outputTransformer = outputTransformer.transpose(0, 1)  # обратно [batch, seq_len, embedding_dim]

    logits = output_layer(outputTransformer)
    loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
    loss.backward()
    optimizer.step()  # обновляем веса
    # После обучения (или внутри цикла, чтобы смотреть динамику)
    with torch.no_grad():
        embedded = embedding_layer(input_ids)
        src = embedded.transpose(0, 1)
        outputTransformer = transformer_encoderLayer(src, src_key_padding_mask=(attention_mask == 0))
        outputTransformer = outputTransformer.transpose(0, 1)
        logits = output_layer(outputTransformer)  # [batch, seq_len, vocab_size]

        # Берём самый вероятный токен для каждого положения
        predicted_token_ids = torch.argmax(logits, dim=-1)  # [batch, seq_len]

        # Переводим индексы обратно в текст
        predicted_text = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
        print("Predicted text:", predicted_text[0])

print(f"Epoch [{epoch + 1}/{epochNum}] — Loss: {loss.item():.4f}")
torch.save({
    'embedding_state': embedding_layer.state_dict(),
    'transformer_state': transformer_encoderLayer.state_dict(),
    'output_state': output_layer.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'epoch': epochmy
}, "model_checkpoint.pt")
