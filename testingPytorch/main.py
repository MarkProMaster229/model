import torch
import torch.nn as nn
import torch.optim as optim

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
probs = torch.softmax(logitOutputLayer, dim=-1)# softmax по словарю

print(probs)
pred_ids = torch.argmax(probs, dim=-1)
print(pred_ids)
pred_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in pred_ids]
#просто сопоставить самый вероятный токен
def decode_tokens(tokens):
    text = ""
    for t in tokens:
        if t.startswith("##"):
            text += t[2:]  # убираем ##
        else:
            text += " " + t
    return text.strip()


sentence = decode_tokens(pred_tokens[0])
print(sentence)
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
optimizer.zero_grad()
loss.backward()
optimizer.step()# обновляем веса
print("Loss:", loss.item())