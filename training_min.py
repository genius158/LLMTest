# minimal_working.py
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os

def create_text_file():
    """创建文本文件，完全避免数据集处理问题"""
    text_content = """
机器学习是人工智能的重要分支。
深度学习使用神经网络进行模式识别。
自然语言处理让计算机理解人类语言。
大语言模型在文本生成方面很强大。
模型微调可以适应特定任务需求。
人工智能技术正在快速发展。
神经网络模型需要大量数据进行训练。
预训练语言模型从海量文本中学习。
注意力机制提高模型处理长文本能力。
Transformer架构是现代NLP的基础。
""" * 100  # 重复创建足够内容
    
    # 保存到文件
    with open('train_data.txt', 'w', encoding='utf-8') as f:
        f.write(text_content)
    
    print(f"📁 创建训练文件，大小: {len(text_content)} 字符")

def main():
    """使用最稳定、最简单的方法"""
    print("🚀 最小化工作版本")
    print("=" * 40)
    
    # 1. 创建训练数据文件
    create_text_file()
    
    # 2. 使用最稳定的模型和tokenizer
    print("🔧 加载GPT-2模型...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ 设置pad_token")
    
    # 3. 使用TextDataset（最稳定的方式）
    print("📊 创建TextDataset...")
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="train_data.txt",
        block_size=128  # 序列长度
    )
    
    # 4. 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言建模
    )
    
    # 5. 训练参数（最简配置）
    training_args = TrainingArguments(
        output_dir="./minimal_output",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        prediction_loss_only=True,
    )
    
    # 6. 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # 7. 训练
    print("🎯 开始训练...")
    trainer.train()
    
    # 8. 保存
    trainer.save_model()
    print("✅ 训练完成! 模型保存到: ./minimal_output")
    
    # 清理
    if os.path.exists("train_data.txt"):
        os.remove("train_data.txt")
        print("🧹 清理临时文件")

if __name__ == "__main__":
    main()