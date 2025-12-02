
# test_tuned_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_tuned_model():
    """测试微调后的模型"""
    model_path = "./tuned_model"
    
    try:
        # 加载微调后的模型
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ 微调模型加载成功")
        
        # 测试文本生成
        test_prompts = [
            "机器学习是",
            "深度学习模型",
            "自然语言处理",
            "大语言模型可以"
        ]
        
        for prompt in test_prompts:
            print(f"\n提示: '{prompt}'")
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # 生成文本
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"生成: {generated}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_tuned_model()
