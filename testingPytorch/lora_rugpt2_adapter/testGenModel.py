# test_generation.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


MODEL_PATH = "/home/chelovek/PycharmProjects/model/finalCut/"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Папка модели не найдена: {MODEL_PATH}")


print("Загрузка модели и токенизатора...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)



input_text = "а вы помните что было вчера ? я вот да"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=40)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Результат генерации:", result)
