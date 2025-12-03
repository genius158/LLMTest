import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset
import json
from datasets import Dataset

model_id = "./models/Qwen3-1.7B"
output_dir = "./ultra_safe_model"

# --- 2. MODEL VE TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cpu"
)

# 配置	参数量	训练速度	效果	适用场景
# ["q_proj", "v_proj"]	最小	最快	基础适配	资源有限，简单任务
# ["q_proj", "k_proj", "v_proj"]	中等	中等	更好的注意力调整	需要理解长上下文的任务
# ["q_proj", "k_proj", "v_proj", "o_proj"]	较大	较慢	全面的注意力调整	复杂推理任务
# 所有注意力层+部分MLP层	最大	最慢	最全面的微调	需要深度领域适应的任务
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# --- 4. VERİ FORMATLAMA ---
# Zorunlu system prompt [cite: 131]
system_prompt = "你是一个AI助手"

def formatting_prompts_func(examples):
    messages = []
    for conv in examples["conversations"]:
        messages.append({
            "role": conv["role"],
            "content": conv["content"]
        })
    
    print(f"messages {messages} \n\n")

    # 使用tokenizer的apply_chat_template
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 不进行tokenize，SFTTrainer会处理
        add_generation_prompt=False
    )

    print(f"formatted_text {formatted_text} \n\n")
    return formatted_text

# --- 训练 ---
print("获取数据 ...")
# 使用load_dataset的标准方式
dataset = load_dataset('json', data_files='./lora_identity_minimind.jsonl', split='train')
print(f"Dataset size: {len(dataset)}")
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     per_device_train_batch_size=1,    
#     gradient_accumulation_steps=16,   
#     gradient_checkpointing=True,      
#     learning_rate=2e-4,
#     num_train_epochs=1,
#     logging_steps=10,
#     save_strategy="epoch",
#     fp16=True,
#     report_to="none"
# )
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=2, #训练轮次
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-4, #通用学习率
    save_steps=30,
    save_total_limit=1,
    weight_decay=0.01, 
    logging_steps=30,
    dataloader_pin_memory=False,
    # 禁用可能引起问题的功能
    prediction_loss_only=True,
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    formatting_func=formatting_prompts_func,
)

print("训练结束")
trainer.train()

trainer.model.save_pretrained(output_dir)
print(f"保存到: {output_dir}")