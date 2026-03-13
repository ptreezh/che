# SOUL 自主进化启动指南

> **目标**：永不停止的持续进化，达成Nature/Science发表使命

---

## 一、自动启动机制

### 1.1 会话开始时执行

```bash
# 每次会话开始时，运行：
python .soul/evolution_engine.py status

# 查看下一个任务
python .soul/evolution_engine.py next
```

### 1.2 任务执行流程

```
1. 加载状态 → python .soul/evolution_engine.py status
2. 获取任务 → python .soul/evolution_engine.py next
3. 执行任务 → 根据任务类型执行相应操作
4. 更新状态 → 修改 evolution_state.json
5. 创建检查点 → python .soul/evolution_engine.py checkpoint
```

---

## 二、任务类型映射

| 任务类型 | 执行方式 |
|---------|---------|
| experiment | 运行实验脚本，收集数据 |
| analysis | 执行数据分析，生成报告 |
| validation | 验证数据质量，检查完整性 |
| publication | 更新论文，准备材料 |

---

## 三、当前优先任务

### 高优先级 (立即执行)

1. **auto_001**: 运行验证实验 - 确认多样性修复效果
2. **exp_3**: 完成15代完整实验

### 中优先级 (后续执行)

3. **auto_002**: 跨模型验证
4. **auto_003**: 完善论文

---

## 四、检查点策略

- **时间间隔**：每小时自动创建检查点
- **事件触发**：任务完成后创建检查点
- **错误恢复**：失败后从最近检查点恢复

---

## 五、自进化规则

```yaml
self_evolution_rules:
  task_completion_threshold: 0.8
  auto_generate_next_tasks: true
  prioritize_failed_tasks: true
  max_concurrent_tasks: 3
  context_threshold: 0.7
```

---

## 六、快速命令

| 命令 | 作用 |
|-----|-----|
| `python .soul/evolution_engine.py status` | 查看状态 |
| `python .soul/evolution_engine.py next` | 获取下一任务 |
| `python .soul/evolution_engine.py checkpoint` | 创建检查点 |
| `python .soul/evolution_engine.py advance` | 推进阶段 |

---

## 七、Hooks配置

### 7.1 会话开始 (on_session_start)

```
动作：加载任务队列，检查进度，继续执行
触发：每次新会话开始
```

### 7.2 任务完成 (on_task_complete)

```
动作：更新任务状态，保存检查点，触发下一任务
触发：任务标记为completed时
```

### 7.3 错误处理 (on_error)

```
动作：记录错误，尝试重试，保存状态
触发：任务执行失败时
```

### 7.4 上下文阈值 (on_context_threshold)

```
动作：压缩上下文，保存进度，继续执行
触发：上下文使用超过70%时
```

---

**版本**：2.0.0  
**更新**：2026-03-13
