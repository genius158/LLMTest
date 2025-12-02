# advanced_validation.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging
from pathlib import Path
import json
from typing import List, Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_validation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ModelValidator:
    """模型验证器"""
    
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
    def check_model_files(self) -> bool:
        """检查模型文件完整性"""
        logger.info("📁 检查模型文件完整性...")
        
        required_files = [
            'config.json',
            'pytorch_model.bin',  # PyTorch模型文件
            'tokenizer_config.json',
            'special_tokens_map.json',
            'vocab.json'  # 对于某些tokenizer
        ]
        
        # 可选的文件
        optional_files = [
            'generation_config.json',
            'model.safetensors',  # 安全格式
            'tokenizer.model'  # 对于sentencepiece
        ]
        
        existing_files = os.listdir(self.model_path) if os.path.exists(self.model_path) else []
        
        logger.info(f"模型目录: {self.model_path}")
        logger.info(f"找到 {len(existing_files)} 个文件")
        
        # 检查必要文件
        missing_files = []
        for file in required_files:
            if file not in existing_files:
                # 检查是否有替代文件
                if file == 'pytorch_model.bin' and 'model.safetensors' in existing_files:
                    logger.info("✅ 找到 model.safetensors (替代 pytorch_model.bin)")
                elif file == 'vocab.json' and 'tokenizer.model' in existing_files:
                    logger.info("✅ 找到 tokenizer.model (替代 vocab.json)")
                else:
                    missing_files.append(file)
        
        if missing_files:
            logger.error(f"❌ 缺少必要文件: {missing_files}")
            return False
        
        logger.info("✅ 模型文件完整性检查通过")
        
        # 显示文件大小
        for file in existing_files:
            file_path = os.path.join(self.model_path, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"  {file}: {size_mb:.1f} MB")
        
        return True
    
    def load_model(self) -> bool:
        """加载模型和tokenizer"""
        logger.info(f"🤖 加载模型...")
        
        try:
            # 1. 加载tokenizer
            logger.info("  加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True  # 对于某些自定义模型
            )
            
            # 确保pad_token设置正确
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"  ✅ Tokenizer加载成功")
            logger.info(f"    词汇表大小: {self.tokenizer.vocab_size}")
            logger.info(f"    模型最大长度: {self.tokenizer.model_max_length}")
            
            # 2. 加载模型
            logger.info("  加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # 移动到设备
            self.model = self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            logger.info(f"  ✅ 模型加载成功")
            logger.info(f"    设备: {self.device}")
            logger.info(f"    参数量: {self.model.num_parameters():,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return False
    
    def test_text_generation(self, prompts: List[str] = None) -> Dict[str, Any]:
        """测试文本生成功能"""
        logger.info("🎯 测试文本生成...")
        
        if prompts is None:
            prompts = [
                "介绍一下人工智能",
                "深度学习是什么？",
                "写一首关于春天的诗：",
                "解释一下量子计算：",
                "Python的列表推导式怎么写？"
            ]
        
        results = {}
        
        for i, prompt in enumerate(prompts):
            logger.info(f"\n📝 测试 {i+1}: '{prompt}'")
            
            try:
                # 编码输入
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # 生成参数
                generate_kwargs = {
                    "input_ids": inputs.input_ids,
                    "max_new_tokens": 50,  # 生成50个新token
                    "num_return_sequences": 1,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.9,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                }
                
                # 如果输入有attention_mask，添加
                if "attention_mask" in inputs:
                    generate_kwargs["attention_mask"] = inputs.attention_mask
                
                # 生成文本
                with torch.no_grad():
                    outputs = self.model.generate(**generate_kwargs)
                
                # 解码生成结果
                generated = self.tokenizer.decode(
                    outputs[0], 
                    skip_special_tokens=True
                )
                
                # 计算生成长度
                input_length = len(inputs.input_ids[0])
                generated_length = len(outputs[0])
                new_tokens = generated_length - input_length
                
                logger.info(f"  输入长度: {input_length} tokens")
                logger.info(f"  输出长度: {generated_length} tokens")
                logger.info(f"  生成 {new_tokens} 个新token")
                logger.info(f"  生成结果: {generated}")
                
                results[f"test_{i+1}"] = {
                    "prompt": prompt,
                    "generated": generated,
                    "input_tokens": input_length,
                    "output_tokens": generated_length,
                    "new_tokens": new_tokens,
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"❌ 生成失败: {e}")
                results[f"test_{i+1}"] = {
                    "prompt": prompt,
                    "error": str(e),
                    "success": False
                }
        
        return results
    
    def calculate_perplexity(self, test_texts: List[str], batch_size: int = 1) -> Optional[float]:
        """计算困惑度（Perplexity）"""
        logger.info("📊 计算困惑度...")
        
        if not test_texts:
            logger.warning("⚠️ 没有测试文本，跳过困惑度计算")
            return None
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        try:
            with torch.no_grad():
                for i in range(0, len(test_texts), batch_size):
                    batch_texts = test_texts[i:i+batch_size]
                    
                    # 批量编码
                    inputs = self.tokenizer(
                        batch_texts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=512
                    ).to(self.device)
                    
                    # 前向传播
                    outputs = self.model(
                        **inputs, 
                        labels=inputs.input_ids
                    )
                    
                    # 累计损失和token数
                    batch_loss = outputs.loss.item()
                    batch_tokens = inputs.input_ids.numel()
                    
                    total_loss += batch_loss * batch_tokens
                    total_tokens += batch_tokens
                    
                    if (i // batch_size) % 10 == 0:
                        logger.info(f"  已处理 {min(i+batch_size, len(test_texts))}/{len(test_texts)} 个文本")
            
            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                perplexity = torch.exp(torch.tensor(avg_loss))
                logger.info(f"✅ 平均损失: {avg_loss:.4f}")
                logger.info(f"✅ 困惑度 (PPL): {perplexity:.2f}")
                return perplexity.item()
            else:
                logger.error("❌ 没有有效的token用于计算困惑度")
                return None
                
        except Exception as e:
            logger.error(f"❌ 计算困惑度失败: {e}")
            return None
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """测试内存使用情况"""
        logger.info("💾 测试内存使用...")
        
        try:
            # 获取模型参数数量
            num_params = self.model.num_parameters()
            
            # 估算模型大小（假设float32）
            model_size_mb = (num_params * 4) / (1024 * 1024)  # 4 bytes per float32
            
            # GPU内存信息
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                logger.info(f"  GPU已分配内存: {allocated:.1f} MB")
                logger.info(f"  GPU保留内存: {reserved:.1f} MB")
            
            logger.info(f"  模型参数量: {num_params:,}")
            logger.info(f"  估算模型大小: {model_size_mb:.1f} MB")
            
            return {
                "num_params": num_params,
                "estimated_size_mb": model_size_mb,
                "device": self.device,
            }
            
        except Exception as e:
            logger.error(f"❌ 内存测试失败: {e}")
            return {}
    
    def analyze_model_config(self) -> Dict[str, Any]:
        """分析模型配置"""
        logger.info("⚙️ 分析模型配置...")
        
        try:
            config_path = os.path.join(self.model_path, "config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 提取关键配置
            important_keys = [
                "vocab_size", "hidden_size", "num_hidden_layers",
                "num_attention_heads", "intermediate_size",
                "max_position_embeddings", "model_type"
            ]
            
            config_info = {}
            for key in important_keys:
                if key in config:
                    config_info[key] = config[key]
                    logger.info(f"  {key}: {config[key]}")
            
            return config_info
            
        except Exception as e:
            logger.error(f"❌ 配置分析失败: {e}")
            return {}
    
    def run_comprehensive_validation(self, test_texts: List[str] = None) -> Dict[str, Any]:
        """运行全面验证"""
        logger.info("🔍 开始全面模型验证")
        print("=" * 60)
        
        validation_results = {
            "model_path": self.model_path,
            "device": self.device,
            "checks_passed": [],
            "checks_failed": [],
            "metrics": {}
        }
        
        # 1. 检查文件
        if self.check_model_files():
            validation_results["checks_passed"].append("file_check")
        else:
            validation_results["checks_failed"].append("file_check")
            return validation_results
        
        # 2. 加载模型
        if self.load_model():
            validation_results["checks_passed"].append("model_load")
        else:
            validation_results["checks_failed"].append("model_load")
            return validation_results
        
        # 3. 分析配置
        config_info = self.analyze_model_config()
        validation_results["config"] = config_info
        
        # 4. 测试内存
        memory_info = self.test_memory_usage()
        validation_results["memory"] = memory_info
        
        # 5. 测试生成
        generation_results = self.test_text_generation()
        validation_results["generation"] = generation_results
        
        # 6. 计算困惑度
        if test_texts:
            perplexity = self.calculate_perplexity(test_texts)
            if perplexity:
                validation_results["metrics"]["perplexity"] = perplexity
                validation_results["checks_passed"].append("perplexity_calculation")
            else:
                validation_results["checks_failed"].append("perplexity_calculation")
        
        # 总结
        logger.info("\n" + "=" * 60)
        logger.info("📋 验证总结:")
        logger.info(f"  通过检查: {len(validation_results['checks_passed'])} 项")
        logger.info(f"  失败检查: {len(validation_results['checks_failed'])} 项")
        
        if validation_results["checks_failed"]:
            logger.error(f"❌ 失败项: {validation_results['checks_failed']}")
        
        return validation_results
    
    def save_validation_report(self, results: Dict[str, Any], output_path: str = "validation_report.json"):
        """保存验证报告"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 验证报告已保存到: {output_path}")
        except Exception as e:
            logger.error(f"❌ 保存报告失败: {e}")


def load_test_texts(file_path: str, max_lines: int = 100) -> List[str]:
    """从文件加载测试文本"""
    if not os.path.exists(file_path):
        logger.warning(f"⚠️ 测试文件不存在: {file_path}")
        return []
    
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if line and len(line) > 10:  # 过滤空行和太短的文本
                    texts.append(line)
        
        logger.info(f"✅ 从 {file_path} 加载了 {len(texts)} 条测试文本")
        return texts
    except Exception as e:
        logger.error(f"❌ 加载测试文本失败: {e}")
        return []


def main():
    """主函数"""
    print("🚀 高级模型验证工具")
    print("=" * 60)
    
    # 配置
    model_path = "./ultra_safe_model"  # 修改为您的模型路径
    test_file_path = "./simple_train.txt"  # 用于计算困惑度的测试文件
    
    # 检查模型路径
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型路径不存在: {model_path}")
        logger.info("💡 请检查路径，或运行训练脚本先训练模型")
        return
    
    # 创建验证器
    validator = ModelValidator(model_path)
    
    # 加载测试文本
    test_texts = load_test_texts(test_file_path)
    
    # 运行全面验证
    results = validator.run_comprehensive_validation(test_texts)
    
    # 保存报告
    validator.save_validation_report(results)
    
    # 输出总结
    print("\n" + "=" * 60)
    if not results["checks_failed"]:
        print("🎉 模型验证通过!")
        print(f"📁 模型路径: {model_path}")
        print(f"📊 生成测试: {len(results.get('generation', {}))} 项")
        
        if "perplexity" in results.get("metrics", {}):
            print(f"📈 困惑度: {results['metrics']['perplexity']:.2f}")
    else:
        print("❌ 模型验证失败")
        print(f"失败项: {results['checks_failed']}")
    
    print("=" * 60)


if __name__ == "__main__":
    # 添加命令行参数支持
    import argparse
    
    parser = argparse.ArgumentParser(description="大语言模型验证工具")
    parser.add_argument("--model_path", type=str, default="./ultra_safe_model", 
                       help="模型目录路径")
    parser.add_argument("--test_file", type=str, default="./simple_train.txt",
                       help="测试文本文件路径")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], 
                       default="auto", help="运行设备")
    
    args = parser.parse_args()
    
    # 设置模型路径
    model_path = args.model_path
    test_file_path = args.test_file
    
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 运行主函数
    main()