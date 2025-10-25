# merge_lora_full.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# Параметры
BASE_MODEL_NAME = "ai-forever/rugpt2large"
LORA_PATH = "/home/chelovek/PycharmProjects/model/testingPytorch/lora_rugpt2_adapter"
OUTPUT_PATH = "/home/chelovek/PycharmProjects/model/finalCut"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- 1. Загружаем базовую модель и токенизатор ---
print("Загрузка базовой модели...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# --- 2. Подключаем LoRA ---
print("Подключаем LoRA...")
lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)

# --- 3. Сливаем LoRA в базовую модель ---
print("Слияние LoRA в базовую модель...")
merged_model = lora_model.merge_and_unload()

# --- 4. Сохраняем финальную модель и токенизатор ---
print("Сохраняем финальную модель и токенизатор...")
merged_model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Все файлы сохранены в папке: {OUTPUT_PATH}")

#print("Проверка генерации...")
#input_text = "Пример: 25+13. Ответ:"
#inputs = tokenizer(input_text, return_tensors="pt").to(merged_model.device)
#outputs = merged_model.generate(**inputs, max_length=50)
#result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print("Результат генерации:", result)
