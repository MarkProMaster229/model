import json
import os
import re
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import logging
from google.colab import drive

logging.set_verbosity_error()

drive.mount('/content/drive')

# Путь к датасету и к директории для чекпоинтов
DATASET_PATH = "/content/drive/MyDrive/твойДатасетС_диска"
OUTPUT_DIR = "/content/drive/MyDrive/rugpt2_2ch_output"

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    texts = json.load(f)  # <- прямо загружаем как список строк

# Создаём HuggingFace Dataset
dataset = Dataset.from_dict({"text": texts})



# Загружаем токенизатор и модель
model_name = "ai-forever/rugpt3small_based_on_gpt2"#в последствии меняй на дообученую тобой модель
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# GPT2 не имеет pad_token нужно указать явно
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

# Параметры обучения не забудь добавить счетчик по шагам, мне без них тяжко пришлось))!!!!!
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    num_train_epochs=5,
    per_device_train_batch_size=14,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    logging_steps=100,
    fp16=True,
    seed=42,
    push_to_hub=False,
    report_to=[]  # вот это отключит wandb и другие логгеры что-бы некий api не требовал
)




# тренер
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# работа тренера
trainer.train()

# Финальное сохранение( на диск вместе с чекпоинтами, рекомендую работать с последним чекпоинтом а не с конечной моделью
# дело в том что модель с чекпоинта можно дообучить что иногда важно)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
