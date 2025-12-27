# 认知异质性实验 - 模型安装脚本

## 概述

此目录包含用于安装认知异质性实验所需模型的脚本，支持Linux和Windows操作系统。

## 文件说明

### Linux脚本
- `install_models_linux.sh` - Linux环境下的模型安装脚本
- `MODEL_INSTALLATION_GUIDE.md` - 详细的安装指南

### Windows脚本  
- `install_models_windows.bat` - Windows环境下的模型安装脚本

## 安装步骤

### Linux系统
1. 确保Ollama已安装并运行
2. 给脚本添加执行权限：`chmod +x install_models_linux.sh`
3. 运行脚本：`./install_models_linux.sh`

### Windows系统
1. 确保Ollama已安装并运行
2. 双击运行 `install_models_windows.bat` 或在命令行中运行

## 实验所需模型

### 核心模型
- `gemma:2b` - 评估器模型
- `qwen:7b-chat` - 替代模型
- `llama3:latest` - 替代模型

### 实验配置模型
- `qwen3:4b` - 实验配置指定
- `llama3.2:3b` - 实验配置指定
- `gemma3:latest` - 评估器使用
- `qwen:0.5b` - 默认配置

### 备用模型
- `llama3.2:1b` - 轻量级选项
- `deepseek-coder:6.7b` - 编程模型

## 系统要求

- 至少15GB可用磁盘空间
- 8GB+内存（推荐）
- 稳定的网络连接
- Ollama服务正在运行

## 验证安装

安装完成后，运行以下命令验证：
```bash
ollama list
```

## 运行实验

安装完成后，可以运行认知异质性实验：
```bash
python -c "
from src.che.experimental.cognitive_independence_experiment import run_evolution_experiment, create_diverse_population
results = run_evolution_experiment(
    ecosystem=create_diverse_population(population_size=30), 
    generations=15, 
    population_size=30
)
print('实验完成！')
"
```

## 故障排除

如果遇到问题，请检查：
1. Ollama服务是否正在运行
2. 网络连接是否稳定
3. 磁盘空间是否充足
4. 详细错误信息

更多帮助请参考 `MODEL_INSTALLATION_GUIDE.md`。