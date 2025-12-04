import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import json
from datasets import Dataset

def main():
    model_id = "./models/Qwen3-1.7B"
    output_dir = "./ultra_safe_model"

    # --- 2. MODEL VE TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )

    # 配置	参数量	训练速度	效果	适用场景
    # ["q_proj", "v_proj"]	最小	最快	基础适配	资源有限，简单任务
    # ["q_proj", "k_proj", "v_proj"]	中等	中等	更好的注意力调整	需要理解长上下文的任务
    # ["q_proj", "k_proj", "v_proj", "o_proj"]	较大	较慢	全面的注意力调整	复杂推理任务
    # 所有注意力层+部分MLP层	最大	最慢	最全面的微调	需要深度领域适应的任务
    peft_config = LoraConfig(
        r=16,  # 较小的r值减少内存使用
        lora_alpha=32,
        lora_dropout=0.05,
        # target_modules=["q_proj", "v_proj"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)


    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} | 总参数: {total_params:,} | 百分比: {100 * trainable_params / total_params:.2f}%")

    system_prompt = "你是一个AI助手"

    def formatting_prompts_func(examples):
        messages = []
        messages.append({
            "role": "system",
            "content": "你是一个AI助手"
        })
        for conv in examples["conversations"]:
            messages.append({
                "role": conv["role"],
                "content": conv["content"]
            })
        
        # 使用tokenizer的apply_chat_template
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 不进行tokenize，SFTTrainer会处理
            add_generation_prompt=False
        )

        return formatted_text

    # --- 训练 ---
    print("获取数据 ...")
    # 使用load_dataset的标准方式
    dataset = load_dataset('json', data_files='./lora_identity.jsonl', split='train')
    print(f"Dataset size: {len(dataset)}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=12, #训练轮次
        per_device_train_batch_size=4,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        learning_rate=1e-4, #通用学习率
        save_steps=50,
        save_total_limit=1,
        weight_decay=0.01, 
        logging_steps=50,
        dataloader_num_workers=0,
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

    print("训练开始")
    train_result = trainer.train()

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"保存到: {output_dir}")

    metrics = train_result.metrics
    print("🎉 训练成功完成!")
    print(f"📊 最终损失: {metrics.get('train_loss', 'N/A')}")
       

if __name__ == '__main__':
    main()