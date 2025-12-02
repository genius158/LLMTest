# 完整工具调用实现 - Qwen模型
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os

# 1. 加载模型和分词器
def load_model():
    # model_name = "./models/Qwen3-1.7B"  # 使用对话优化模型
    model_name = "./ultra_safe_model"  # 使用对话优化模型
    print(f"正在加载模型: {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16  # 使用半精度节省显存
        )
        print("✅ 模型和分词器加载成功！")
        return tokenizer, model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return tokenizer, model

# 2. 定义工具
def define_tools():
    tools = [
        {
            "name": "calculator",
            "description": "计算数学表达式的结果（支持加减乘除、括号）",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "待计算的数学表达式，如 '2*(3+4)'"
                    }
                },
                "required": ["expression"]
            }
        },
        {
            "name": "get_weather",
            "description": "查询指定城市的实时天气（模拟接口）",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'"
                    }
                },
                "required": ["city"]
            }
        },
        {
            "name": "search_web",
            "description": "在互联网上搜索最新信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，如 '2025年科技趋势'"
                    }
                },
                "required": ["query"]
            }
        }
    ]
    return tools

# 3. 实现工具执行函数
def execute_tool(tool_name, parameters):
    """执行工具并返回结果"""
    print(f"🔧 执行工具: {tool_name}，参数: {parameters}")
    
    if tool_name == "calculator":
        try:
            # 安全计算表达式（生产环境应使用更安全的方法）
            expression = parameters["expression"]
            # 替换常见的数学符号
            expression = expression.replace('×', '*').replace('÷', '/')
            result = eval(expression)  # 注意：实际应用中应使用安全计算库
            return f"计算结果：{expression} = {result}"
        except Exception as e:
            return f"计算失败：{str(e)}"
    
    elif tool_name == "get_weather":
        city = parameters["city"]
        # 模拟天气数据（实际可调用真实API）
        mock_weather = {
            "北京": "晴，25°C，东南风3级",
            "上海": "多云，28°C，东风2级",
            "广州": "阵雨，30°C，南风4级",
            "深圳": "雷阵雨，29°C，西南风3级",
            "杭州": "阴，26°C，北风2级"
        }
        return mock_weather.get(city, f"未找到{city}的天气数据，请尝试其他城市")
    
    elif tool_name == "search_web":
        query = parameters["query"]
        # 模拟搜索结果（实际可调用搜索引擎API）
        mock_results = {
            "2025年科技趋势": "2025年十大科技趋势：1. 量子计算商业化 2. AI通用智能突破 3. 脑机接口普及...",
            "人工智能发展历史": "人工智能发展史：1956年达特茅斯会议提出AI概念，经历三次浪潮...",
            "最新iPhone发布": "苹果将于2025年9月发布iPhone 17系列，搭载全新A19芯片..."
        }
        return mock_results.get(query, f"未找到关于'{query}'的最新信息")
    
    else:
        return f"未知工具：{tool_name}"

# 4. 带工具调用的对话函数
def chat_with_tools(tokenizer, model, user_input, history=None):
    """带工具调用的对话函数"""
    if history is None:
        history = []
    
    # 获取工具定义
    tools = define_tools()
    
    # 1. 构造系统提示（包含工具定义）
    system_prompt = (
        "你是一个智能助手，可以调用以下工具解决问题：\n"
        f"{json.dumps(tools, ensure_ascii=False, indent=2)}\n"
        "调用工具时需严格按格式返回 JSON：{\"name\": \"工具名\", \"parameters\": {\"参数名\": \"值\"}}，"
        "不要添加其他内容。如果不需要调用工具，直接回答用户问题。"
    )


    # 2. 整理对话历史
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    # print(f"🤔 对话：{messages}")
    

    # 3. 模型生成响应
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # 提取模型响应
    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):], 
        skip_special_tokens=True
    ).strip()
    
    print(f"\n🤖 模型原始响应: {response}")
    
    # 4. 解析模型输出：判断是否需要调用工具
    try:
        # 尝试解析 JSON 格式的工具调用请求
        tool_call = json.loads(response)
        
        # 检查是否是有效的工具调用格式
        if isinstance(tool_call, dict) and "name" in tool_call and "parameters" in tool_call:
            tool_name = tool_call["name"]
            parameters = tool_call["parameters"]
            
            # 执行工具
            tool_result = execute_tool(tool_name, parameters)
            
            # 打印工具执行结果
            print(f"⚙️ 工具执行结果: {tool_result}")
            
            # 将工具调用和结果加入历史
            new_history = history + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response},  # 模型的工具调用请求
                {"role": "tool", "content": tool_result}     # 工具执行结果
            ]
            
            # 递归调用，让模型基于工具结果生成最终回答
            return chat_with_tools(tokenizer, model, "请根据工具结果回答用户问题", new_history)
    
    except (json.JSONDecodeError, TypeError):
        # 模型直接生成回答（无需工具）
        pass
    
    # 返回最终回答和更新后的历史
    updated_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ]
    
    return response, updated_history

# 5. 主函数 - 测试工具调用
def main():
    # 加载模型
    tokenizer, model = load_model()
    
    # 测试对话
    test_questions = [
        "介绍一下 严贤炜",
        "计算 (15 + 3) * 2 的结果",
        "what's the weather like today in beijing？",
        "搜索一下2025年科技趋势",
        "人工智能的发展历史是怎样的？",
    ]
    
    history = []
    
    print("\n" + "="*50)
    print("🚀 开始工具调用测试")
    print("="*50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 测试 {i}: {question}")
        print("-"*50)
        
        try:
            answer, history = chat_with_tools(tokenizer, model, question, history)
            print(f"💬 AI回答: {answer}")
        except Exception as e:
            print(f"❌ 处理问题时出错: {e}")
        
        print("-"*50)
    
    print("\n✅ 测试完成！")

if __name__ == "__main__":
    main()