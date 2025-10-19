#testing here slm model, and working mcp-tool

#dowload model slm, get

# Use a pipeline as a high-level helper

import torch
from transformers import AutoTokenizer, Phi3ForCausalLM

model_path = "/home/chelovek/models/Phi-3.5-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = Phi3ForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": "cpu"},        # на CPU
    attn_implementation="eager"    # вместо flash-attention
)

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=16)

print(tokenizer.decode(outputs[0]))

#further