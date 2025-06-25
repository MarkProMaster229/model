import json

import drive
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, DataCollatorForLanguageModeling, TrainerCallback
from transformers import logging

from model.methods.training import get_training_args,get_trainer

logging.set_verbosity_error()

drive.mount('/content/drive')

# Путь к датасету и к директории для чекпоинтов
DATASET_PATH = "/content/drive/MyDrive/сюдаТвойДатасет"
OUTPUT_DIR = "/content/drive/MyDrive/rugpt2_Modal_output"

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    texts = json.load(f)  # <- прямо загружаем как список строк

# Создаём HuggingFace Dataset
dataset = Dataset.from_dict({"text": texts})

# Загружаем токенизатор и модель
model_name = "/content/drive/MyDrive/model"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# GPT2 не имеет pad_token
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Токенизация
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Коллатор
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
# Параметры обучения cm класс training.py

training_args = get_training_args(OUTPUT_DIR)
print("тест работы ")

# === Кастомный Callback для вывода шагов
class StepPrinterCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % training_args.logging_steps == 0:
            print(f"Шаг: {state.global_step}")

# Тренер
trainer =get_trainer(model,training_args,tokenized_dataset,tokenizer,data_collator,StepPrinterCallback)
# Обучение
trainer.train()

# Сохранение модели
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
