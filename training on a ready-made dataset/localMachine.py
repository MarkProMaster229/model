#pip install transformers datasets torch

import os
import json
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    logging
)
from datasets import Dataset

# Настройки логирования
logging.set_verbosity_error()

# Пути
MODEL_PATH = "./model2"  # Укажите путь к локальной модели
DATA_PATH = "./russian_instructions.jsonl"  # Путь к jsonl-файлу
OUTPUT_DIR = "./rugpt2_local_output"  # Куда сохранять модель после обучения

# Читаем JSONL-файл
texts = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        text = f"Вопрос: {item['question']}\nОтвет: {item['answer']}"
        texts.append({"text": text})

# Создание датасета
dataset = Dataset.from_list(texts)

# Загрузка токенизатора и модели
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# Установка pad_token
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
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Аргументы обучения
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=2,
    num_train_epochs=5,
    per_device_train_batch_size=15,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    logging_steps=100,
    fp16=False,  #FP16(для карт Nvidia), если нет поддержки (например, на CPU)
    seed=42,
    push_to_hub=False,
    report_to=[]
)

print("Начинаем обучение...")

# Callback для логирования
class StepPrinterCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % training_args.logging_steps == 0:
            print(f"Шаг: {state.global_step}")

# Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[StepPrinterCallback()]
)

# Обучение
trainer.train()

# Сохранение модели и токенизатора
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
