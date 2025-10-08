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
src = embedded.transpose(0, 1) #теперь seq_len затем batch затемem bedding_dim - это требование трнсформера

transformerLayer = nn.TransformerEncoderLayer(
    d_model=embedding_dim,
    nhead=12,#ахринеть как все просто!!!! это типо я щас 10 голов внимания имею!!!
    dim_feedforward=2048, # внутренняя размерность FFN
    dropout=0.1, #Dropout - отключает 10% нейронов случайно при обучении
    batch_first=False

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
#просто сапоставить самый вероятный токен
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
#обратное распространение

referense = tokenizer(["Чел, иди в роблокс играй!"])
input_ids = encoded['input_ids']
target_ids = input_ids.clone()
torch.save(attention_mask, "input_ids.pt")


criterion = nn.CrossEntropyLoss()

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