from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import json
from datasets import Dataset
#тупо качаем
print("пошло поехало")

model_name = "ai-forever/rugpt2large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=228,
    lora_alpha=228,
    target_modules=[
        "c_attn",  # внимание
        "c_proj",  # выход внимания
        "c_fc",  # вход MLP
        "c_proj"  # выход MLP (не путай с attention.c_proj)
    ],
)

# оборачиваем модель LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

input_text = ("Пример: 25+13")
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)

with open("datasetCalc.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

data = {"text": texts}
dataset = Dataset.from_dict(data)


dataset = Dataset.from_dict(data)
def tokenize(batch):
    encodings = tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=64
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings
tokenized = dataset.map(tokenize, batched=True)

#просто обучение LoRa
training_args = TrainingArguments(
    output_dir="./lora_rugpt2",
    per_device_train_batch_size=2,
    learning_rate=1e-4,
    num_train_epochs=7,
    logging_steps=1,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()
model.save_pretrained("lora_rugpt2_adapter")