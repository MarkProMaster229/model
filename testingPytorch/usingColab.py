# -----------------------------
# Установка зависимостей
# -----------------------------
!pip install -q bitsandbytes transformers accelerate --upgrade

import torch
import shutil
from google.colab import drive
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

drive.mount('/content/drive')


model_dir = "/content/drive/MyDrive/Phi-3.5-mini-int4"  # измени под свой путь


tokenizer = AutoTokenizer.from_pretrained(model_dir)
print("Токенизатор загружен")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)


model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    quantization_config=bnb_config
)
print("Модель загружена")

messages = [{"role": "user", "content": "сколько будет 156+361"}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

print("Генерация...")
outputs = model.generate(**inputs, max_new_tokens=40)

print("Результат:")
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
