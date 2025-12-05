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
        device_map="cpu",
        trust_remote_code=True,
        dtype=torch.float32  # 使用半精度节省显存
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ 模型和分词器加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit()

# 4. 使用您的原始代码（已修复）
# 输入文本
messages = [
    # {'role': 'system', 'content': '你是AI助手，能够全面的给出问题的回复'},
    {'role': 'user', 'content': "多角度全面的介绍一下严贤炜"}
]

 # 3. 模型生成响应
text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
# 分词并转换为模型输入
inputs = tokenizer("多角度全面的介绍一下严贤炜", return_tensors="pt").to(model.device)

# 生成参数
# 更合理的生成参数配置
generate_kwargs = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# 生成输出
outputs = model.generate(**inputs, **generate_kwargs)

# 解码并打印结果
# 清理输出结果
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n生成结果:")
print(response)