from transformers import LlamaForCausalLM, AutoTokenizer
import torch

model = LlamaForCausalLM.from_pretrained(
    "inference-llama3-8b",
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("inference-llama3-8b", use_fast=True)

inputs = tokenizer("What is 23 + 58?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
