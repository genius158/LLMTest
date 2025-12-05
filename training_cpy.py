# ultimate_fix.py
import torch
from transformers import (
    AutoTokenizer, 
    TextDataset,
    LineByLineTextDataset,
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from inspect_tokenized_dataset import TokenizedDatasetInspector

from peft import LoraConfig, get_peft_model, TaskType
import json
from typing import List, Dict, Any
import os
import logging
import numpy as np
import pandas as pd


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
train_path = "./simple_train.txt"
model_name ="./models/Qwen3-1.7B"
# model_name ="./ultra_safe_model"

def prepare_line_based_dataset(tokenizer, file_path: str, max_length: int = 128) -> Dataset:
    """准备基于行的训练数据集"""
    logger.info(f"📚 准备行级训练数据: {file_path}")
        
    # 加载文本文件（每行一个样本）
    dataset = load_dataset('text', data_files={'train': train_path})['train']
        
    # Tokenize函数
    def tokenize_function(examples):
        logger.info(f"📚 tokenize_function: {examples}")
        # 对每行独立tokenize
        tokenized = tokenizer(
            examples['text'],
            truncation=True,      # 截断到max_length
            padding="max_length", # 填充到max_length，方便批处理
            max_length=128,
            return_tensors="pt"   # 返回PyTorch张量
        )
        return tokenized
        
    # 应用tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10,
        remove_columns=dataset.column_names,
        desc="Tokenizing lines for LM"
    )
        
    logger.info(f"✅ 数据准备完成: {len(tokenized_dataset)} 个训练样本")
    return tokenized_dataset


def get_data_from_cvs(tokenizer) -> Dataset:
    try:
        # 1. 读取CSV文件
        df = pd.read_csv("./tran_data.csv")
        print(f"成功加载CSV文件，共 {len(df)} 行数据")
        print(f"数据列: {df.columns.tolist()}")
        
        # 2. 检查必要列
        if 'prompt' not in df.columns or 'response' not in df.columns:
            raise ValueError("CSV文件必须包含'prompt'和'response'列")
        
        # 3. 创建训练文本
        train_texts = []
        for _, row in df.iterrows():
            # 创建格式化的对话文本
            text = f"用户: {row['prompt']}\n助手: {row['response']}"
            train_texts.append({"text": text})
        
        # 4. 创建Dataset
        dataset = Dataset.from_list(train_texts)
        print(f"创建Dataset，共 {len(dataset)} 条样本")
        
        # 5. Tokenize函数
        def tokenize_function(examples):
            """
            分词处理函数
            """
            # Tokenize文本
            tokenized = tokenizer(
                examples["text"],
                truncation=True,      # 截断到max_length
                padding="max_length", # 填充到max_length，方便批处理
                max_length=128,
                return_tensors="pt"   # 返回PyTorch张量
            )
            
            # 对于语言模型训练，labels通常是input_ids的副本
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        # 6. 应用分词函数
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=10,
            remove_columns=["text"]  # 移除原始文本列，节省内存
        )
        
        print(f"分词完成，数据集大小: {len(tokenized_dataset)}")
        return tokenized_dataset
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_path}")
        raise
    except Exception as e:
        print(f"处理数据时发生错误: {e}")
        raise


# 1. 加载和准备数据
def load_and_format_data(file_path):
    """读取JSONL文件并转换为数据集"""
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line.strip()))
    
    # 转换为huggingface数据集格式
    dataset = Dataset.from_list(data)
    return dataset

# 2. 数据格式化函数
def format_conversations(example):
    """格式化对话数据为模型输入格式"""
    # Qwen3的对话格式
    formatted_text = ""
    for i, message in enumerate(example["conversations"]):
        role = message['role']
        content = message['content']
        
        if role == 'user':
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == 'assistant':
            formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == 'system':
            formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
    
    logger.info(f"formatted_text: {formatted_text}")
    return {"text": formatted_text}


def ultra_safe_tokenize_and_train():
    """超安全的tokenization和训练流程"""
    logger.info("🚀 开始超安全训练流程")
    print("=" * 60)
    
    output_dir = "./ultra_safe_model"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 使用最稳定的模型
        logger.info(f"🤖 加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # 2. 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA秩
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],  # 标准Transformer模块
            bias="none",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("✅ 模型加载成功")
        
        logger.info("📊 创建TextDataset...")
        
        # 加载并处理数据
        print("加载训练数据...")
        dataset = load_and_format_data("lora_identity_minimind.jsonl")
        
        # 应用格式化函数
        dataset = dataset.map(format_conversations)
        
        # 划分训练集和验证集
        print("分割数据集...")
        dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
        
        # 检查分割结果
        print(f"训练集大小: {len(dataset_dict['train'])}")
        print(f"测试集大小: {len(dataset_dict['test'])}")
        
        # 7. 分词函数
        def tokenize_function(examples):
            """分词函数"""
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors=None  # 返回普通字典而不是tensors
            )
        
        # 8. 分别对训练集和测试集进行分词
        print("分词处理...")
        
        # 分词训练集
        tokenized_train = dataset_dict["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "conversations"]  # 移除原始列
        )
        
        # 分词测试集
        tokenized_test = dataset_dict["test"].map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "conversations"]  # 移除原始列
        )
        
        print(f"分词后训练集特征: {tokenized_train.features}")
        print(f"分词后测试集特征: {tokenized_test.features}")
    
        # 9. 检查数据集是否为空
        if len(tokenized_train) == 0:
            print("错误: 训练集为空!")
            return None, None
        
        # 5. 配置数据整理器（关键：让DataCollator处理labels）
        logger.info("🔧 配置DataCollator...")
        # 喂给模型训练的标准批次数据字典（通常包含 input_ids， attention_mask， labels）
        # DataCollatorForLanguageModeling可以自动处理
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 因果语言建模
        )
        
        # 6. 训练参数（最简配置）
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
            remove_unused_columns=False, #删除无效的组
            dataloader_pin_memory=False,
            # 禁用可能引起问题的功能
            prediction_loss_only=True,
        )
        
        # 7. 创建训练器
        logger.info("🎯 创建训练器...")
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
        )
        
        # 8. 开始训练
        logger.info("🔥 开始训练...")
        print("=" * 50)
        
        train_result = trainer.train()
        
        # 9. 保存模型
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        metrics = train_result.metrics
        logger.info("🎉 训练成功完成!")
        logger.info(f"📊 最终损失: {metrics.get('train_loss', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始----")

    # 运行超安全训练
    success = ultra_safe_tokenize_and_train()
    
    if success:
        print("\n 训练成功")
    else:
        print("\n 训练失败")