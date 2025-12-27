# 认知异质性实验 - 模型安装指南

## Linux环境模型安装

### 1. 系统要求
- Linux操作系统 (Ubuntu, CentOS, Debian等)
- Ollama已安装并运行
- 至少15GB可用磁盘空间
- 稳定的网络连接

### 2. 安装Ollama (如果未安装)

```bash
# Ubuntu/Debian
curl -fsSL https://ollama.ai/install.sh | sh

# 或者手动下载
wget https://github.com/jmorganca/ollama/releases/latest/download/ollama-linux-amd64.tgz
tar -xzf ollama-linux-amd64.tgz
sudo mv ollama /usr/local/bin/
```

### 3. 启动Ollama服务

```bash
# 方法1: 前台运行
ollama serve

# 方法2: 后台运行
nohup ollama serve > ollama.log 2>&1 &

# 方法3: 使用systemd (推荐)
sudo systemctl enable ollama
sudo systemctl start ollama
```

### 4. 使用安装脚本

```bash
# 1. 给脚本添加执行权限
chmod +x install_models_linux.sh

# 2. 运行安装脚本
./install_models_linux.sh
```

### 5. 实验所需模型

安装脚本将安装以下模型：

#### 核心模型
- `gemma:2b` - 用于评估器和基础任务
- `qwen:7b-chat` - 替代qwen:0.5b，用于标准型智能体
- `llama3:latest` - 替代llama3.2:3b，用于批判型智能体

#### 实验配置模型
- `qwen3:4b` - 实验配置中指定的主要模型
- `llama3.2:3b` - 实验配置中指定的模型
- `gemma3:latest` - 实验代码中使用的评估模型
- `qwen:0.5b` - 默认配置中指定的模型

#### 备用模型
- `llama3.2:1b` - 轻量级模型选项
- `deepseek-coder:6.7b` - 编程相关任务模型

### 6. 验证安装

安装完成后，可以验证模型是否正确安装：

```bash
# 列出所有模型
ollama list

# 测试模型功能
ollama run gemma:2b
ollama run qwen:7b-chat
```

### 7. 运行实验

安装完成后，您可以运行认知异质性实验：

```bash
# 小规模测试
python test_local_models.py

# 完整实验
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

### 8. 故障排除

#### Ollama服务未运行
```bash
# 检查服务状态
ps aux | grep ollama

# 重启服务
pkill ollama
ollama serve
```

#### 模型下载失败
- 检查网络连接
- 尝试使用代理
- 手动下载特定模型：`ollama pull gemma:2b`

#### 磁盘空间不足
- 清理不需要的模型：`ollama rm model_name`
- 检查可用空间：`df -h`

### 9. 注意事项

- 模型下载可能需要较长时间（30分钟到数小时）
- 某些模型可能需要8GB+内存
- 建议在下载期间保持网络连接稳定
- 可以根据需要调整模型列表

### 10. 高级配置

如需自定义模型列表，可以编辑 `install_models_linux.sh` 文件中的 `models` 数组。

---
**注意**: 此脚本适用于Linux环境。Windows用户请参考 `install_models_windows.bat` (如存在)。