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


def debug_labels_structure(labels, name="labels"):
    """深度调试labels结构"""
    logger.info(f"🔍 调试{name}结构:")
    logger.info(f"  类型: {type(labels)}")
    
    if isinstance(labels, list):
        logger.info(f"  长度: {len(labels)}")
        if labels:
            first_item = labels[0]
            logger.info(f"  第一个元素类型: {type(first_item)}")
            
            if isinstance(first_item, list):
                logger.error("❌ 检测到嵌套列表!")
                if first_item and isinstance(first_item[0], list):
                    logger.error("❌ 检测到双重嵌套列表!")
                logger.info(f"    嵌套示例: {labels[:2]}")
            elif isinstance(first_item, (int, np.integer)):
                logger.info("✅ 是整数列表 - 正确格式")
            else:
                logger.warning(f"⚠️ 包含 {type(first_item)} 类型元素")
    else:
        logger.warning(f"⚠️ 不是列表: {type(labels)}")


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
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("✅ 模型加载成功")
        
        logger.info("📊 创建TextDataset...")
        # train_dataset = LineByLineTextDataset(
        #     tokenizer=tokenizer,
        #     file_path=train_path,
        #     block_size=128  # 序列长度
        # )
        # train_dataset = prepare_line_based_dataset(tokenizer,train_path,128)
        train_dataset = get_data_from_cvs(tokenizer)

          # 2. 基础查看器
        inspector = TokenizedDatasetInspector(tokenizer)
        
        # 3. 基础查看
        inspector.basic_inspection(train_dataset, num_samples=3)
        
        # 4. 统计分析
        stats = inspector.statistical_analysis(train_dataset)
        
        # 5. 解码显示
        inspector.decode_and_display(train_dataset, num_samples=2)
        

        logger.info(f"✅ 数据集创建成功，样本数: {len(train_dataset)}")
        
        # 4. 验证数据集结构
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info("🔍 验证数据集样本结构:")
            
            # 检查样本类型和结构
            logger.info(f"  样本类型: {type(sample)}")
            if hasattr(sample, 'keys'):
                logger.info(f"  样本键: {list(sample.keys())}")
            else:
                # TextDataset返回的是字典
                if isinstance(sample, dict):
                    for key, value in sample.items():
                        logger.info(f"  {key}: 类型={type(value)}")
                        if hasattr(value, '__len__'):
                            logger.info(f"   长度: {len(value)}")
                            debug_labels_structure(value, key)
                else:
                    logger.info(f"  样本值类型: {type(sample)}")
                    if hasattr(sample, '__len__'):
                        logger.info(f"  样本长度: {len(sample)}")
                        debug_labels_structure(sample, "样本")
        
        # 5. 配置数据整理器（关键：让DataCollator处理labels）
        logger.info("🔧 配置DataCollator...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 因果语言建模
        )
        

        # 6. 训练参数（最简配置）
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=4, #训练轮次
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-5, #通用学习率
            save_steps=10,
            save_total_limit=1,
            weight_decay=0.01, 
            logging_steps=10,
            remove_unused_columns=False, #删除无效的组
            dataloader_pin_memory=False,
            # 禁用可能引起问题的功能
            prediction_loss_only=True,  # 让DataCollator处理
        )
        
        # 7. 创建训练器
        logger.info("🎯 创建训练器...")
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
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
    print("🚀 终极修复方案启动")
    print("=" * 60)

    # create_simple_text_file()
    
    # 运行超安全训练
    success = ultra_safe_tokenize_and_train()
    
    if success:
        print("\n 训练成功")
    else:
        print("\n 训练失败")