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
from google.colab import drive

logging.set_verbosity_error()

# Монтируем Google Drive
drive.mount('/content/drive')

# Пути
MODEL_PATH = "/content/drive/MyDrive/model2"
DATA_PATH = "/content/drive/MyDrive/russian_instructions.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/rugpt2_Modal_output"

# Читаем jsonl вручную
texts = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        # Объединяем вопрос и ответ в одну строку для обучения
        text = f"Вопрос: {item['question']}\nОтвет: {item['answer']}"
        texts.append({"text": text})

# Создаём HuggingFace Dataset
dataset = Dataset.from_list(texts)

# Загружаем токенизатор и модель
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# GPT2 не имеет pad_token, назначаем eos_token
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

# Параметры обучения
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
    fp16=True,
    seed=42,
    push_to_hub=False,
    report_to=[]
)

print("Начинаем обучение...")

# Callback для вывода шагов
class StepPrinterCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % training_args.logging_steps == 0:
            print(f"Шаг: {state.global_step}")

# Создаём тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[StepPrinterCallback()]
)

# Запускаем обучение
trainer.train()

# Сохраняем модель и токенизатор
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
