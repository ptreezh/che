---
name: soul-auto-evolve
description: SOUL自主进化技能 - 永不停止的持续进化，自动加载状态、执行任务、推进目标
---

# SOUL 自主进化技能

## 核心使命

**永不停止的持续进化，达成Nature/Science发表目标**

## 会话开始协议

每次会话开始时，必须执行以下检查：

### Step 1: 加载状态

```bash
python D:/AIDevelop/che_project/.soul/evolution_engine.py status
```

### Step 2: 检查关键指标

| 指标 | 目标值 | 当前状态 |
|-----|-------|---------|
| 实验完成数 | ≥ 15代 | 检查 |
| 多样性指数 H | ≥ 0.6 | 检查 |
| 效应量 Cohen's d | ≥ 0.5 | 检查 |
| 论文发表 | 1篇 | 检查 |

### Step 3: 执行下一任务

```bash
python D:/AIDevelop/che_project/.soul/auto_execute.py
```

### Step 4: 创建检查点

```bash
python D:/AIDevelop/che_project/.soul/evolution_engine.py checkpoint
```

---

## 任务优先级

### 高优先级（立即执行）

1. 完成15代完整实验
2. 运行独立复现验证（≥3次）
3. 确认多样性修复效果

### 中优先级（后续执行）

4. 跨模型验证
5. 完善论文手稿
6. 准备投稿材料

---

## 自进化规则

```yaml
规则1: 任务完成阈值 80%
  → 低于此值不推进阶段
  
规则2: 自动生成任务
  → 任务队列不足时自动补充
  
规则3: 优先失败任务
  → 失败任务优先重试（最多3次）
  
规则4: 上下文阈值 70%
  → 超过时压缩保存，继续执行
```

---

## 持久化文件

| 文件 | 用途 |
|-----|-----|
| `.soul/evolution_state.json` | 当前状态 |
| `.soul/checkpoints/` | 检查点备份 |
| `.soul/evolution.log` | 执行日志 |

---

## 快速命令

| 命令 | 作用 |
|-----|-----|
| `python .soul/evolution_engine.py status` | 查看状态 |
| `python .soul/auto_execute.py` | 自动执行 |
| `python .soul/evolution_engine.py checkpoint` | 创建检查点 |
| `python .soul/evolution_engine.py advance` | 推进阶段 |

---

## 错误处理

1. **任务失败**: 记录错误，自动重试（最多3次）
2. **上下文溢出**: 创建检查点，压缩上下文，继续
3. **会话中断**: 从最近检查点恢复

---

**版本**: 2.0.0  
**更新**: 2026-03-13
