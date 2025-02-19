import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_name = "Qwen/Qwen2.5-0.5B"

# Load the model (use float32 for CPU)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map=None  # Do not use device_map, since we are only using CPU
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prompt message
prompt = "Give me a short introduction to large language model."

# Prepare the chat history
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# If `apply_chat_template` is not available, manually create the input prompt
text = ""
for message in messages:
    role = message["role"]
    content = message["content"]
    if role == "user":
        text += f"User: {content}\n"
    elif role == "system":
        text += f"System: {content}\n"

# Tokenize the text
model_inputs = tokenizer(text, return_tensors="pt")

# Ensure inputs are moved to the CPU
model_inputs = {key: value.to("cpu") for key, value in model_inputs.items()}

# Generate text from the model
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512  # Limit for generation
)

# Decode the output (skip special tokens)
response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Print the response
print(response)
print(model)
