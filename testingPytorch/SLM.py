#testing here slm model, and working mcp-tool

#dowload model slm, get

# Use a pipeline as a high-level helper

from transformers import pipeline

pipe = pipeline("text-generation", model="microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
#further