from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型和分词器
model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 输入提示
prompt = "Once upon a time, in a faraway land, there was a magical creature called the"

# 将输入文本转换为模型需要的格式
inputs = tokenizer(prompt, return_tensors="pt")
# inputs["input_ids"]
# inputs["attention_mask"]
# 生成文本
# max_new_tokens 控制生成的额外 token 数量
# temperature 控制生成的随机性，值越小生成越稳定
generated_ids = model.generate(**inputs, max_new_tokens=100, temperature=0.7)

# 将生成的 token 转换回文本
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 打印结果
print("Generated Text:")
print(generated_text)