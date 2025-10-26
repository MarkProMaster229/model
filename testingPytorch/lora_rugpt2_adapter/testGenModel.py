# test_generation.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import subprocess


MODEL_PATH = "/home/chelovek/Документы/modelExp/finalyTestingModel"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Папка модели не найдена: {MODEL_PATH}")


print("Загрузка модели и токенизатора...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)



input_text = "сколько будет 11+5 "
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=40)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
subprocess.run(["/home/chelovek/Документы/modelExp/model/Tool/calc", result])
print("Результат генерации:", result)
