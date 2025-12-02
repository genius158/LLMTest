# basic_validation.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def basic_model_validation(model_path):
    """基础模型验证：加载和简单生成测试"""
    logger.info("🔍 开始基础模型验证")
    print("=" * 50)
    
    # 1. 检查模型文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型路径不存在: {model_path}")
        return False
    
    required_files = ['pytorch_model.bin', 'config.json', 'tokenizer.json']
    existing_files = os.listdir(model_path)
    logger.info(f"📁 模型文件: {existing_files}")
    
    # 2. 加载模型和tokenizer
    try:
        logger.info("🤖 加载微调后的模型...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()  # 设置为评估模式
        
        logger.info(f"✅ 模型加载成功，设备: {device}")
        logger.info(f"✅ Tokenizer词汇表大小: {tokenizer.vocab_size}")
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return False
    
    # 3. 测试文本生成
    test_prompts = [
        "严贤炜是",
        "胡容是",
        "自然语言处理",
        "大语言模型可以",
        "大语言模型在文本生成方面优秀"
    ]
    
    logger.info("🎯 测试文本生成...")
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\n📝 测试 {i+1}: '{prompt}'")
        
        try:
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # 生成文本
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + 20,  # 生成20个新token
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask
                )
            
            # 解码生成结果
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"  生成: {generated}")
            
        except Exception as e:
            logger.error(f"❌ 生成失败: {e}")
            continue
    
    return True

def calculate_perplexity(model, tokenizer, test_texts, device="cpu"):
    """计算困惑度（Perplexity）"""
    logger.info("📊 计算困惑度...")
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in test_texts:
            try:
                # 编码文本
                inputs = tokenizer(text, return_tensors="pt").to(device)
                
                # 前向传播
                outputs = model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                
                # 累计损失和token数
                total_loss += loss.item() * len(inputs.input_ids[0])
                total_tokens += len(inputs.input_ids[0])
                
            except Exception as e:
                logger.warning(f"⚠️ 跳过文本 '{text[:30]}...': {e}")
                continue
    
    if total_tokens > 0:
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        logger.info(f"✅ 困惑度: {perplexity:.2f}")
        return perplexity.item()
    else:
        logger.error("❌ 无法计算困惑度")
        return None

if __name__ == "__main__":
    model_path = "./ultra_safe_model"  # 修改为您的模型路径
    
    # 基础验证
    success = basic_model_validation(model_path)
    
    if success:
        logger.info("🎉 基础验证通过!")
    else:
        logger.error("❌ 基础验证失败")