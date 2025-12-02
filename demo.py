# 1. 导入必要库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# 3. 加载分词器和模型（使用正确的模型ID）
# model_name = "./models/Qwen3-1.7B"  # 完整模型ID（组织名+模型名）

model_name = "./ultra_safe_model"  # 完整模型ID（组织名+模型名）
try:
    # 加载分词器（必须加 trust_remote_code=True）
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        fix_mistral_regex=True
    )
    
    # 加载模型（自动分配设备）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16  # 使用半精度节省显存
    )
    print("✅ 模型和分词器加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit()

# 4. 使用您的原始代码（已修复）
# 输入文本
prompt = "严贤炜的社交圈子"

# 分词并转换为模型输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成参数
generate_kwargs = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

# 生成输出
outputs = model.generate(**inputs, **generate_kwargs)

# 解码并打印结果
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n生成结果:")
print(response)