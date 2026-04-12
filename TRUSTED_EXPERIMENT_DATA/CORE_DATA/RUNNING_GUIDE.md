# CHE补充实验 - 试点实验运行指南

**版本**: 1.0  
**最后更新**: 2026-04-12  
**预计运行时间**: 2-4小时（试点）

---

## 快速开始（Quick Start）

### 1. 环境检查（5分钟）

```powershell
# 检查Python版本
python --version  # 应显示 3.10+

# 检查Ollama服务
ollama list       # 应显示5个模型

# 检查工作目录
pwd               # 应显示 D:\AIDevelop\che_project
```

### 2. 运行试点实验

```powershell
# 切换到实验目录
cd TRUSTED_EXPERIMENT_DATA\CORE_DATA

# 运行试点实验
python pilot_experiment.py
```

### 3. 查看结果

```powershell
# 检查输出目录
ls pilot_results\

# 查看汇总
cat pilot_results\pilot_summary.json
```

---

## 详细步骤

### 步骤1: 预运行准备（15分钟）

#### 1.1 阅读协议

```powershell
# 打开并阅读实验协议
notepad EXPERIMENTAL_PROTOCOL.md
```

**重点阅读**: 
- 第2节：实验设计（6个组的定义）
- 第3节：方法学细节（参数设置）
- 第4节：统计方法（假设检验）

#### 1.2 检查质量控制清单

```powershell
# 打开检查清单
notepad QUALITY_CONTROL_CHECKLIST.md
```

**逐一确认**:
- [ ] 所有6项"实验前检查"已完成
- [ ] 所有模型已安装并可调用
- [ ] 磁盘空间充足（>10GB）

#### 1.3 创建输出目录

```powershell
# 创建结果目录
mkdir -Force pilot_results

# 验证创建成功
Test-Path pilot_results  # 应返回 True
```

---

### 步骤2: 启动实验（2-4小时）

#### 2.1 启动Ollama服务

```powershell
# 确保Ollama在运行
# 如果未运行，打开新PowerShell窗口并执行：
ollama serve

# 验证服务状态
ollama ps  # 应显示服务状态
```

#### 2.2 运行实验脚本

```powershell
# 在CORE_DATA目录下
python pilot_experiment.py 2>&1 | Tee-Object pilot_run.log
```

**预期输出**:
```
======================================================================
CHE补充实验 - 试点研究
开始时间: 2026-04-12 14:30:00
======================================================================

============================================================
运行实验组 A: deepseek + single
============================================================
Gen  1: Mean=7.832, ModelH=0.000, RoleH=0.000
Gen  2: Mean=7.891, ModelH=0.000, RoleH=0.000
...

✓ 组 A 结果已保存: pilot_results\group_A_result.json
```

#### 2.3 监控运行

**打开新PowerShell窗口监控进度**:

```powershell
# 实时查看输出
tail -f pilot_run.log

# 或查看当前运行的代
Get-Content pilot_results\group_A_result.json | ConvertFrom-Json | Select-Object -ExpandProperty generations | Select-Object generation, mean_score | Format-Table
```

**预期进度**:
- 每组约20-30分钟
- 6组总计约2-4小时

---

### 步骤3: 实验后验证（15分钟）

#### 3.1 检查输出文件

```powershell
# 列出所有结果文件
ls pilot_results\*.json

# 应显示:
# group_A_result.json
# group_B_result.json
# group_C_result.json
# group_D_result.json
# group_E_result.json
# group_F_result.json
# pilot_summary.json
```

#### 3.2 验证数据完整性

```powershell
# 验证JSON文件可解析
python -c "
import json
import sys
for group in ['A', 'B', 'C', 'D', 'E', 'F']:
    try:
        with open(f'pilot_results/group_{group}_result.json', 'r') as f:
            data = json.load(f)
        print(f'✓ Group {group}: {len(data[\"generations\"])} generations')
    except Exception as e:
        print(f'✗ Group {group}: ERROR - {e}')
        sys.exit(1)
print('\n✓ All files validated successfully!')
"
```

#### 3.3 查看结果汇总

```powershell
# 使用Python查看汇总
python -c "
import json
with open('pilot_results/pilot_summary.json', 'r') as f:
    summary = json.load(f)

print('='*60)
print('试点实验结果汇总')
print('='*60)
print(f\"{'组':<6} {'模型类型':<15} {'角色配置':<12} {'最终分数':<10}\")
print('-'*60)
for g in summary['groups']:
    print(f\"{g['group_id']:<6} {g['model_type']:<15} {g['role_config']:<12} {g['final_score']:<10.3f}\")
"
```

---

### 步骤4: 数据分析（30分钟）

#### 4.1 基本统计分析

```powershell
# 运行统计分析脚本（创建analysis.py）
python analysis.py
```

**创建analysis.py**:
```python
import json
import numpy as np
from scipy import stats

# 读取所有组数据
groups = {}
for group_id in ['A', 'B', 'C', 'D', 'E', 'F']:
    with open(f'pilot_results/group_{group_id}_result.json', 'r') as f:
        groups[group_id] = json.load(f)

# 提取最终分数
final_scores = {
    gid: data['generations'][-1]['mean_score']
    for gid, data in groups.items()
}

# 组定义
group_definitions = {
    'A': ('deepseek', 'single'),
    'B': ('deepseek', 'multi'),
    'C': ('glm4', 'single'),
    'D': ('glm4', 'multi'),
    'E': ('mixed', 'single'),
    'F': ('mixed', 'multi')
}

# 关键对比
comparisons = [
    ('F', 'A', '总异质性效应'),
    ('F', 'E', '角色多样性贡献'),
    ('F', 'B', '模型多样性贡献'),
    ('B', 'A', 'DeepSeek纯角色效应'),
    ('E', 'A', '混合模型纯模型效应'),
]

print('='*70)
print('试点实验关键对比分析')
print('='*70)

for g1, g2, desc in comparisons:
    score1 = final_scores[g1]
    score2 = final_scores[g2]
    diff = score1 - score2
    pct = (diff / score2) * 100
    print(f'{desc}')
    print(f'  {g1} vs {g2}: {score1:.3f} - {score2:.3f} = {diff:+.3f} ({pct:+.1f}%)')
    print()

# 2x2因子分析（简化）
print('='*70)
print('因子效应分析')
print('='*70)

# 计算边际均值
model_means = {
    'deepseek': np.mean([final_scores['A'], final_scores['B']]),
    'glm4': np.mean([final_scores['C'], final_scores['D']]),
    'mixed': np.mean([final_scores['E'], final_scores['F']])
}

role_single = np.mean([final_scores['A'], final_scores['C'], final_scores['E']])
role_multi = np.mean([final_scores['B'], final_scores['D'], final_scores['F']])

print(f'模型类型主效应:')
for model, mean in model_means.items():
    print(f'  {model}: {mean:.3f}')

print(f'\n角色多样性主效应:')
print(f'  单角色: {role_single:.3f}')
print(f'  多角色: {role_multi:.3f}')
print(f'  差异: {role_multi - role_single:+.3f}')
```

#### 4.2 生成可视化

```powershell
# 使用matplotlib生成图表（创建visualization.py）
python visualization.py
```

---

### 步骤5: 决策与报告（30分钟）

#### 5.1 评估成功标准

根据实验结果回答：

| 标准 | 评估 | 是否通过 |
|------|------|----------|
| 完成率 6/6 | ___/6 | [ ] |
| 运行时间 <24h | ___h | [ ] |
| 结果合理性 | 描述: | [ ] |
| 方向一致性 | 描述: | [ ] |

#### 5.2 决策

**选项1**: 继续正式实验（n=15/组）  
**选项2**: 修订协议后重新试点  
**选项3**: 终止实验（结果不支持假设）

**决策理由**: 

```
[在此记录决策理由]
```

#### 5.3 生成报告

```powershell
# 创建实验报告
cat > pilot_report.md << 'EOF'
# CHE补充实验 - 试点研究报告

**日期**: 2026-04-12  
**执行者**: [姓名]  
**协议版本**: 1.0

## 执行摘要

[简要总结实验结果]

## 结果详情

[插入数据表格和图表]

## 成功标准评估

[逐项评估]

## 决策建议

[推荐下一步行动]

## 附件

- [ ] 原始数据文件（6个JSON）
- [ ] 运行日志（pilot_run.log）
- [ ] 统计分析输出
- [ ] 可视化图表
EOF
```

---

## 故障排除

### 问题1: Ollama无法连接

**症状**: `Error: Connection refused`

**解决**:
```powershell
# 1. 检查Ollama进程
Get-Process | Where-Object {$_.Name -like "*ollama*"}

# 2. 重启Ollama
pkill ollama
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden

# 3. 等待10秒后重试
Start-Sleep -Seconds 10
ollama list
```

### 问题2: 模型加载失败

**症状**: `Error: model not found`

**解决**:
```powershell
# 安装缺失的模型
ollama pull llama3:latest
ollama pull qwen2.5:7b
ollama pull deepseek-r1:8b
ollama pull glm4:9b
ollama pull minicpm-v:latest
```

### 问题3: 内存不足

**症状**: `MemoryError` 或系统卡顿

**解决**:
```powershell
# 1. 关闭其他应用
# 2. 降低并发数（修改脚本中的并发参数）
# 3. 分批运行实验组
```

### 问题4: 运行中断

**症状**: 实验中途停止

**解决**:
```powershell
# 检查已完成哪些组
ls pilot_results\group_*.json

# 如果部分完成，可修改脚本仅运行未完成组
# 或从断点继续（需修改代码支持checkpoint恢复）
```

---

## 联系与支持

**技术问题**: [记录问题及解决方案]  
**科学问题**: [咨询统计顾问或领域专家]  
**紧急情况**: [联系项目负责人]

---

## 附录

### A. 常用命令速查

```powershell
# 查看帮助
python pilot_experiment.py --help

# 仅运行特定组（需修改脚本支持）
python pilot_experiment.py --groups A,B,C

# 指定随机种子
python pilot_experiment.py --seed 123

# 指定输出目录
python pilot_experiment.py --output custom_results/
```

### B. 性能优化建议

1. **关闭Windows Defender实时保护**（实验期间）
2. **禁用Windows更新**（实验期间）
3. **使用SSD存储**结果文件
4. **保持电源连接**，禁用休眠

### C. 数据备份命令

```powershell
# 实验完成后备份
Compress-Archive -Path pilot_results\* -DestinationPath pilot_results_backup_$(Get-Date -Format "yyyyMMdd").zip

# 复制到云盘（如果需要）
Copy-Item pilot_results_backup_*.zip D:\OneDrive\Experiments\
```

---

**开始实验前请确认**: 我已阅读并理解本指南的所有步骤。

**签名**: _________________ **日期**: _________________
