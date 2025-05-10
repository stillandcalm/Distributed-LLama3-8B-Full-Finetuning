from transformers import LlamaForCausalLM, AutoTokenizer
import torch

# 1) load your trained model (fp16) directly from the DeepSpeed output
model = LlamaForCausalLM.from_pretrained(
    "output/checkpoint-10",
    device_map="auto",
    torch_dtype=torch.float16,
)

# 2) load its tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "output/checkpoint-10",
    use_fast=True,
)

# 3) re-save everything for inference in a single folder
model.save_pretrained(
    "inference-llama3-8b",
    safe_serialization=True,      # writes a single .safetensors (or sharded) file + index
)
tokenizer.save_pretrained("inference-llama3-8b")
