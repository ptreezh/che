#!/bin/bash

# 认知异质性实验 - 模型安装脚本 (Linux)
# 用于安装实验所需的所有Ollama模型

set -e  # 遌置遇到错误时退出

echo "==========================================="
echo "认知异质性实验 - 模型安装脚本 (Linux)"
echo "==========================================="

# 检查Ollama是否已安装
if ! command -v ollama &> /dev/null; then
    echo "错误: Ollama未安装"
    echo "请先安装Ollama: https://github.com/jmorganca/ollama"
    exit 1
fi

echo "Ollama已安装，版本信息:"
ollama -v

# 检查Ollama服务是否正在运行
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "警告: Ollama服务未运行"
    echo "请先启动Ollama服务: ollama serve"
    echo "或者在新终端中运行: nohup ollama serve > ollama.log 2>&1 &"
    exit 1
fi

echo "Ollama服务正在运行"

# 实验所需模型列表
models=(
    "gemma:2b"           # 本地已支持，用于评估器
    "qwen:7b-chat"       # 本地已支持，替代qwen:0.5b
    "llama3:latest"      # 本地已支持，替代llama3.2:3b
    "llama3.2:1b"        # 本地已支持
    "deepseek-coder:6.7b" # 本地已支持
    "qwen3:4b"           # 实验配置中需要
    "llama3.2:3b"        # 实验配置中需要
    "gemma3:latest"      # 实验代码中需要
    "qwen:0.5b"          # 默认配置中需要
)

echo ""
echo "开始安装实验所需模型..."

# 安装模型函数
install_model() {
    local model=$1
    echo ""
    echo "检查模型: $model"
    
    # 检查模型是否已存在
    if ollama list | grep -q "$model"; then
        echo "✓ 模型 $model 已存在"
    else
        echo "正在下载模型: $model"
        if ollama pull "$model"; then
            echo "✓ 成功安装模型: $model"
        else
            echo "✗ 安装模型失败: $model"
            return 1
        fi
    fi
}

# 逐个安装模型
failed_models=()
for model in "${models[@]}"; do
    if ! install_model "$model"; then
        failed_models+=("$model")
    fi
done

echo ""
echo "==========================================="
echo "安装完成报告"
echo "==========================================="

if [ ${#failed_models[@]} -eq 0 ]; then
    echo "✅ 所有模型安装成功！"
    echo ""
    echo "可用模型列表:"
    ollama list
    echo ""
    echo "现在您可以运行认知异质性实验了！"
    echo ""
    echo "示例命令:"
    echo "  python -c \"from src.che.experimental.cognitive_independence_experiment import run_evolution_experiment, create_diverse_population; results = run_evolution_experiment(ecosystem=create_diverse_population(population_size=30), generations=15, population_size=30); print('实验完成！')\""
else
    echo "⚠️  以下模型安装失败:"
    for failed_model in "${failed_models[@]}"; do
        echo "  - $failed_model"
    done
    echo ""
    echo "请检查网络连接或模型名称是否正确，然后重试。"
    exit 1
fi

echo ""
echo "==========================================="
echo "实验环境准备就绪！"
echo "==========================================="