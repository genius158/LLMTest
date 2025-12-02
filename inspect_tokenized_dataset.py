# inspect_tokenized_dataset.py
from datasets import Dataset
from transformers import AutoTokenizer
import logging
from typing import List, Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizedDatasetInspector:
    """tokenized_dataset 内容查看器"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def basic_inspection(self, tokenized_dataset: Dataset, num_samples: int = 5):
        """基础查看方法"""
        print("🔍 tokenized_dataset 基础查看")
        print("=" * 50)
        
        # 1. 基本信息
        print("📊 数据集基本信息:")
        print(f"   样本数量: {len(tokenized_dataset):,}")
        print(f"   特征列: {tokenized_dataset.column_names}")
        
        # 2. 数据结构
        if len(tokenized_dataset) > 0:
            sample = tokenized_dataset[0]
            print(f"   样本结构: {type(sample)}")
            if isinstance(sample, dict):
                print(f"   样本键: {list(sample.keys())}")
                for key, value in sample.items():
                    if hasattr(value, 'shape'):
                        print(f"     {key}: 形状 {value.shape}")
                    elif isinstance(value, list):
                        print(f"     {key}: 长度 {len(value)}")
                    else:
                        print(f"     {key}: {type(value)}")
        
        # 3. 查看前几个样本
        print(f"\n📄 前 {num_samples} 个样本:")
        for i in range(min(num_samples, len(tokenized_dataset))):
            self._print_sample_details(tokenized_dataset, i, f"样本 {i+1}")
        
        return tokenized_dataset
    
    def _print_sample_details(self, dataset: Dataset, index: int, title: str = "样本"):
        """打印样本详情"""
        sample = dataset[index]
        print(f"\n🎯 {title} (索引 {index}):")
        
        if isinstance(sample, dict):
            for key, value in sample.items():
                if key == 'input_ids' and hasattr(self, 'tokenizer'):
                    # 解码 token IDs
                    try:
                        decoded_text = self.tokenizer.decode(value, skip_special_tokens=True)
                        print(f"   {key}: {len(value)} tokens")
                        print(f"      内容: {decoded_text[:100]}{'...' if len(decoded_text) > 100 else ''}")
                    except:
                        print(f"   {key}: {value} (无法解码)")
                elif isinstance(value, list):
                    print(f"   {key}: 长度 {len(value)}")
                    if len(value) > 0 and isinstance(value[0], (int, float)):
                        print(f"      前5个值: {value[:5]}{'...' if len(value) > 5 else ''}")
                else:
                    print(f"   {key}: {value}")
        else:
            print(f"   {sample}")
    
    def statistical_analysis(self, tokenized_dataset: Dataset):
        """统计分析"""
        print("\n📈 数据集统计分析")
        print("=" * 40)
        
        if len(tokenized_dataset) == 0:
            print("❌ 数据集为空")
            return
        
        stats = {}
        
        # token 长度分析
        if 'input_ids' in tokenized_dataset.column_names:
            lengths = [len(sample['input_ids']) for sample in tokenized_dataset]
            stats['token_lengths'] = {
                'min': min(lengths),
                'max': max(lengths),
                'mean': sum(lengths) / len(lengths),
                'std': (sum((x - sum(lengths)/len(lengths))**2 for x in lengths) / len(lengths))**0.5
            }
            
            print("📏 Token 长度统计:")
            print(f"   最短: {stats['token_lengths']['min']} tokens")
            print(f"   最长: {stats['token_lengths']['max']} tokens")
            print(f"   平均: {stats['token_lengths']['mean']:.1f} tokens")
            print(f"   标准差: {stats['token_lengths']['std']:.1f} tokens")
            
            # 长度分布
            length_bins = [0, 10, 20, 50, 100, 200, 500, 1000, float('inf')]
            length_distribution = {}
            for i in range(len(length_bins)-1):
                count = sum(1 for length in lengths if length_bins[i] <= length < length_bins[i+1])
                if count > 0:
                    length_distribution[f"{length_bins[i]}-{length_bins[i+1]}"] = count
            
            print("   长度分布:")
            for range_str, count in length_distribution.items():
                percentage = count / len(lengths) * 100
                print(f"     {range_str}: {count} 样本 ({percentage:.1f}%)")
        
        return stats
    
    def decode_and_display(self, tokenized_dataset: Dataset, num_samples: int = 3):
        """解码并显示原始文本"""
        print("\n🔤 解码显示原始文本")
        print("=" * 40)
        
        if 'input_ids' not in tokenized_dataset.column_names:
            print("❌ 数据集不包含 input_ids")
            return
        
        for i in range(min(num_samples, len(tokenized_dataset))):
            sample = tokenized_dataset[i]
            input_ids = sample['input_ids']
            
            # 解码
            try:
                decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                original_text = decoded_text
                
                print(f"\n📖 样本 {i+1}:")
                print(f"   Token数量: {len(input_ids)}")
                print(f"   内容: {original_text}")
                
                # 显示特殊token
                if 'special_tokens_mask' in sample:
                    special_count = sum(sample['special_tokens_mask'])
                    print(f"   特殊token数量: {special_count}")
                
            except Exception as e:
                print(f"❌ 解码失败: {e}")