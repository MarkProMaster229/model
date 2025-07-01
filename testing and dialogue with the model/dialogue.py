# pip install transformers torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# === Настройки модели ===
MODEL_NAME = "/home/chelovek/PycharmProjects/PythonProject6/checkpoint-5400"

# === Параметры генерации ===
generation_params = {
    "temperature": 0.2,
    "max_length": 100,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "do_sample": True,  # True = sampling, False = greedy или beam search
    "num_beams": 1,     # >1 — beam search
    "eos_token_id": None,  # заполним ниже
    "pad_token_id": None   # заполним ниже
}

print("Загрузка модели и токенизатора...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()

# === Устройство ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Спец-токены ===
generation_params["eos_token_id"] = tokenizer.eos_token_id
generation_params["pad_token_id"] = tokenizer.eos_token_id  # GPT-2 не имеет pad_token

chat_history = ""

print("Чат с GPT-2. Напиши 'exit' чтобы выйти.")

while True:
    user_input = input("Ты: ")
    if user_input.strip().lower() == "exit":
        break

    # Формируем историю диалога
    chat_history += f"User: {user_input}\nGPT-2:"

    # Токенизируем
    input_ids = tokenizer.encode(chat_history, return_tensors="pt").to(device)

    # Контроль длины истории (если вдруг превысит лимит модели)
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, -1024:]

    # Генерация
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + generation_params["max_length"],
            temperature=generation_params["temperature"],
            top_k=generation_params["top_k"],
            top_p=generation_params["top_p"],
            repetition_penalty=generation_params["repetition_penalty"],
            no_repeat_ngram_size=generation_params["no_repeat_ngram_size"],
            do_sample=generation_params["do_sample"],
            num_beams=generation_params["num_beams"],
            pad_token_id=generation_params["pad_token_id"],
            eos_token_id=generation_params["eos_token_id"]
        )

    # Декодируем только ответ модели
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Добавляем в историю
    chat_history += generated_text + "\n"

    print(f"GPT-2: {generated_text.strip()}")
