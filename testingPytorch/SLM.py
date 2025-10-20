import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


model_dir = "/home/chelovek/PycharmProjects/PythonProject11/model/testingPytorch/Phi-3.5-mini-int4/"


tokenizer = AutoTokenizer.from_pretrained(model_dir)
print("ток")


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
print("модель загружена ")


messages = [{"role": "user", "content": "Who are you?"}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
print("генер ")
outputs = model.generate(**inputs, max_new_tokens=40)
print("последовательность  ")
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
