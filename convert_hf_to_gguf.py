# convert_hf_to_gguf.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama

def convert_hf_to_gguf():
    # 输入和输出路径
    hf_model_path = "./merged_model"
    gguf_output_path = "./converted_model.gguf"
    
    print("正在加载 HuggingFace 模型...")
    
    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    
    print("模型加载完成，开始转换...")
    
    # 保存为临时格式以便转换
    temp_path = "./temp_hf_model"
    model.save_pretrained(temp_path)
    tokenizer.save_pretrained(temp_path)
    
    print("转换完成，模型已保存为中间格式")
    return temp_path

if __name__ == "__main__":
    temp_path = convert_hf_to_gguf()
    print(f"中间模型保存在: {temp_path}")