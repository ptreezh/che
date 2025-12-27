@echo off
REM 认知异质性实验 - 模型安装脚本 (Windows)
REM 用于安装实验所需的所有Ollama模型

echo ===========================================
echo 认知异质性实验 - 模型安装脚本 (Windows)
echo ===========================================

REM 检查Ollama是否已安装
where ollama >nul 2>nul
if errorlevel 1 (
    echo 错误: Ollama未安装
    echo 请先安装Ollama: https://github.com/jmorganca/ollama
    pause
    exit /b 1
)

echo Ollama已安装，版本信息:
ollama -v

REM 检查Ollama服务是否正在运行
tasklist /FI "IMAGENAME eq ollama-windows-amd64.exe" 2>nul | find /I /N "ollama-windows-amd64.exe">nul
if "%ERRORLEVEL%"=="0" (
    echo Ollama服务正在运行
) else (
    echo 警告: Ollama服务未运行
    echo 请先启动Ollama服务: ollama serve
    echo 或者在新命令行窗口中运行: ollama serve
    pause
    exit /b 1
)

REM 实验所需模型列表
set models=gemma:2b qwen:7b-chat llama3:latest llama3.2:1b deepseek-coder:6.7b qwen3:4b llama3.2:3b gemma3:latest qwen:0.5b

echo.
echo 开始安装实验所需模型...

REM 安装模型
for %%m in (%models%) do (
    echo.
    echo 检查模型: %%m
    ollama list | findstr /C:"%%m" >nul
    if errorlevel 1 (
        echo 正在下载模型: %%m
        ollama pull %%m
        if errorlevel 1 (
            echo 错误: 安装模型失败: %%m
        ) else (
            echo 成功安装模型: %%m
        )
    ) else (
        echo 模型 %%m 已存在
    )
)

echo.
echo ===========================================
echo 安装完成报告
echo ===========================================

echo 可用模型列表:
ollama list

echo.
echo 现在您可以运行认知异质性实验了！
echo.
echo 示例命令:
echo   python -c "from src.che.experimental.cognitive_independence_experiment import run_evolution_experiment, create_diverse_population; results = run_evolution_experiment(ecosystem=create_diverse_population(population_size=30), generations=15, population_size=30); print('实验完成！')"
echo.

pause