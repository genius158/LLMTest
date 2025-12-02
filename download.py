from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型名称（以实际 HF 仓库名为准）
model_name = "Qwen/Qwen3-1.7B"  # 或 "Qwen/Qwen1.5-4B-Chat"（若为对话优化版）
save_path = './models/Qwen3-1.7B'

print(f'下载模型: {model_name}')

# 加载分词器（Qwen 通常使用自定义分词器，需 trust_remote_code）
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,  # 关键：允许加载模型自定义的代码
    padding_side="left",     # 对话场景常用左填充
)

# 加载模型（自动选择设备，优先 GPU）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       # 自动分配设备（GPU/CPU）
    trust_remote_code=True,  # 必须，因 Qwen 有自定义模型结构
    torch_dtype="auto",      # 自动选择精度（如 float16/bfloat16）
)

print(f'保存到: {save_path}')
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
