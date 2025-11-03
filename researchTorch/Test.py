from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import json
from datasets import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
print("Test Configuration")
#loading free model, and testing it here

 # Load model directly
#from transformers import AutoTokenizer, AutoModelForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("katanemo/Arch-Router-1.5B")
#model = AutoModelForCausalLM.from_pretrained("katanemo/Arch-Router-1.5B")
#messages = [
#    {"role": "user", "content": "Who are you?"},
#]
#inputs = tokenizer.apply_chat_template(
#	messages,
#	add_generation_prompt=True,
#	tokenize=True,
#	return_dict=True,
#	return_tensors="pt",
#).to(model.device)

#outputs = model.generate(**inputs, max_new_tokens=40)
#print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
ds = load_dataset("MarkProMaster229/synthetic_dataset", split="train",streaming=True )
model_using = "katanemo/Arch-Router-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_using)
model = AutoModelForCausalLM.from_pretrained(model_using,torch_dtype = "auto", device_map="auto")

promt = "привет, расскажи кто ты"
#text in tenzor 
tenzor = tokenizer(promt, return_tensors="pt").to(model.device)

output = model.generate(
    **tenzor,
    max_new_tokens = 50,
    temperature = 0.1
)
exi = tokenizer.decode(output[0])
print(exi)

#pass in LoRA
for name, module in model.named_modules():
    print(name, type(module))

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r = 25,
    lora_alpha=35,
    target_modules=[
        "q_proj",     # attention
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",  # MLP
        "up_proj"
    ]
    
)
model = get_peft_model(model,lora_config)
model.print_trainable_parameters()
dataset = ds
def tokenize(example):
    text = f"Вопрос: {example['input']}\nОтвет: {example['target']}"
    tokenized = tokenizer(text, truncation=True, padding="max_length")
    
    input_len = len(tokenizer(f"Вопрос: {example['input']}\nОтвет: ")[0])
    labels = tokenized["input_ids"].copy()
    labels[:input_len] = -100  # ignore prompt
    tokenized["labels"] = labels
    
    return tokenized

def data_generator():
    for ex in dataset:
        yield tokenize(ex)
        
dataloader = DataLoader(data_generator(), batch_size=2)

tokenized_dataset = dataset.map(tokenize, batched=True)

training_arg = TrainingArguments(
    output_dir="./lora_rugpt2",
    per_device_train_batch_size=2,
    learning_rate=1e-4,
    num_train_epochs=7,
    logging_steps=1,
    save_strategy="epoch",
    
)
trainer = Trainer(
    model=model,
    args=training_arg,
    train_dataset=dataloader,
)
trainer.train()

model.save_pretrained("lora_rugpt2_adapter")