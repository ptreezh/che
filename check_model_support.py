"""
模型支持检查脚本
用于验证本地Ollama模型是否支持实验需求
"""

import ollama
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def check_model_availability():
    """检查本地模型可用性"""
    print("检查本地Ollama模型可用性...")
    
    # 从配置文件中提取需要的模型
    required_models = [
        "qwen:0.5b",      # 默认配置
        "gemma:2b",       # 默认配置
        "qwen3:4b",       # 实验配置
        "llama3.2:3b",    # 实验配置
        "deepseek-coder:6.7b",  # 实验配置
        "gemma3:latest",  # 实验代码
        "qwen:7b-chat",   # 本地可用
        "llama3:latest",  # 本地可用
        "llama3.2:1b"    # 实验配置
    ]
    
    # 获取本地可用模型
    try:
        local_models_response = ollama.list()
        # 检查响应格式并适配 - 处理 Ollama 的 ListResponse 类型
        if hasattr(local_models_response, 'models'):
            # 如果是 Ollama 的 ListResponse 对象
            local_models = [model.name for model in local_models_response.models]
        elif isinstance(local_models_response, dict) and 'models' in local_models_response:
            local_models = [model['name'] for model in local_models_response['models']]
        elif isinstance(local_models_response, list):
            local_models = [model['name'] for model in local_models_response]
        else:
            print(f"未知的响应格式: {type(local_models_response)}")
            # 使用之前从命令行获取的模型列表
            local_models = [
                "deepseek-coder:6.7b-instruct",
                "llama3:instruct",
                "qwen:7b-chat",
                "mistral:instruct",
                "deepseek-coder:6.7b",
                "gemma:2b",
                "llama3.2:1b",
                "llama3:latest"
            ]
        print(f"本地可用模型: {local_models}")
    except Exception as e:
        print(f"无法获取本地模型列表: {e}")
        # 手动列出已知的本地模型
        local_models = [
            "deepseek-coder:6.7b-instruct",
            "llama3:instruct",
            "qwen:7b-chat",
            "mistral:instruct",
            "deepseek-coder:6.7b",
            "gemma:2b",
            "llama3.2:1b",
            "llama3:latest"
        ]
        print(f"使用已知本地模型列表: {local_models}")
    
    print("\n模型支持检查结果:")
    print("-" * 50)
    
    supported_models = []
    unsupported_models = []
    
    for model in required_models:
        if model in local_models:
            print(f"✓ {model:20} - 支持")
            supported_models.append(model)
        else:
            # 检查是否有相似模型
            similar_models = [lm for lm in local_models if model.split(':')[0] in lm]
            if similar_models:
                print(f"△ {model:20} - 不支持，但有相似模型: {similar_models}")
                unsupported_models.append((model, similar_models))
            else:
                print(f"✗ {model:20} - 不支持")
                unsupported_models.append((model, []))
    
    print(f"\n总结:")
    print(f"支持的模型数量: {len(supported_models)}")
    print(f"不支持的模型数量: {len(unsupported_models)}")
    
    if unsupported_models:
        print(f"\n需要安装的模型:")
        for model, similar in unsupported_models:
            if not similar:  # 完全不支持的模型
                print(f"  - {model}")
    
    return supported_models, unsupported_models

def suggest_model_substitutions():
    """建议模型替代方案"""
    print("\n模型替代建议:")
    print("-" * 30)
    print("qwen:0.5b   -> qwen:7b-chat (本地已支持)")
    print("qwen3:4b    -> qwen:7b-chat (本地已支持)")
    print("llama3.2:3b  -> llama3:latest (本地已支持)")
    print("llama3.2:1b  -> llama3:latest (本地已支持)")
    print("gemma3:latest -> gemma:2b (本地已支持)")
    print("\n注意: 替代模型可能在性能上略有差异，但功能相似")

def test_model_functionality(model_name):
    """测试模型基本功能"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Hello, please respond with 'OK' only."}]
        )
        print(f"✓ {model_name} - 功能正常: {response['message']['content'][:20]}...")
        return True
    except Exception as e:
        print(f"✗ {model_name} - 功能异常: {str(e)}")
        return False

if __name__ == "__main__":
    print("认知异质性实验 - 模型支持检查")
    print("=" * 50)
    
    supported, unsupported = check_model_availability()
    suggest_model_substitutions()
    
    print(f"\n测试部分支持模型的功能:")
    print("-" * 30)
    
    # 测试一些本地支持的模型
    test_models = ["gemma:2b", "qwen:7b-chat", "llama3:latest", "deepseek-coder:6.7b"]
    for model in test_models:
        if model in [m for m in supported]:
            test_model_functionality(model)