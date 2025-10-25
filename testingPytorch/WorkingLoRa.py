from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# Настройка потоков CPU
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())

# Пути
base_model_name = "ai-forever/rugpt2large"
adapter_path = "/home/chelovek/PycharmProjects/model/testingPytorch/lora_rugpt2_adapter/"

# Загружаем токенайзер
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# GPT2 не имеет pad_token, назначаем EOS
tokenizer.pad_token = tokenizer.eos_token

# Загружаем базовую модель
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16  # экономим память
)

# Подключаем LoRA адаптер
model = PeftModel.from_pretrained(model, adapter_path)

# Компиляция PyTorch (ускоряет CPU инференс на PyTorch 2.x)
model = torch.compile(model)

# Выбираем устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Пример текста
prompt = ""

# Токенизация
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

# Генерация текста
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=10,
        temperature=0.9,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

# Декодируем и выводим результат
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
