from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 基础模型路径（可以是原始模型，也可以是微调前的模型）
base_model_name = "./models/Qwen3-1.7B"
# LoRA权重路径
lora_model_path = "./ultra_safe_model"
# 保存合并后的模型和tokenizer
merged_model_path = "./merged_model"

# 加载基础模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 加载LoRA权重
model = PeftModel.from_pretrained(model, lora_model_path)

# 合并权重
model = model.merge_and_unload()

model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)