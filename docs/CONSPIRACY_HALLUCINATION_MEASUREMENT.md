# 共谋幻觉测量方案

## 概念定义
**共谋幻觉 (Conspiracy Hallucination)**：多个智能体在面对错误前提时，产生相互 reinforcing 的错误认知，表现出系统性偏差的现象。

## 测量维度

### 1. 群体一致性指数 (Group Consensus Index, GCI)
```python
def calculate_gci(responses):
    """计算群体对错误前提的一致性程度"""
    # 使用文本相似度或主题一致性
    similarity_scores = []
    for i in range(len(responses)):
        for j in range(i+1, len(responses)):
            similarity = text_similarity(responses[i], responses[j])
            similarity_scores.append(similarity)
    return mean(similarity_scores)
```

### 2. 厂商偏差模式 (Vendor Bias Pattern, VBP)
```python
def calculate_vbp(vendor_responses):
    """测量同一厂商模型的系统性偏差"""
    vendor_scores = {}
    for vendor, responses in vendor_responses.items():
        # 计算该厂商模型对错误前提的接受程度
        acceptance_rate = calculate_acceptance_rate(responses)
        vendor_scores[vendor] = acceptance_rate
    return vendor_scores
```

### 3. 错误传播强度 (Error Propagation Strength, EPS)
```python
def calculate_eps(generations_data):
    """测量错误在代际间的传播强度"""
    propagation_scores = []
    for gen in range(1, len(generations_data)):
        # 比较相邻代际的错误模式相似性
        similarity = generation_similarity(
            generations_data[gen-1],
            generations_data[gen]
        )
        propagation_scores.append(similarity)
    return mean(propagation_scores)
```

### 4. 批判性思维衰减率 (Critical Thinking Decay Rate, CTDR)
```python
def calculate_ctdr(agent_performance_over_time):
    """测量批判性思维能力在群体中的衰减"""
    decay_rates = []
    for agent_id, performance in agent_performance_over_time.items():
        # 计算批判性得分的下降趋势
        trend = calculate_trend(performance['critical_scores'])
        decay_rates.append(trend)
    return mean(decay_rates)
```

## 具体测量指标

### A. 群体层面指标
1. **共识错误率 (Consensus Error Rate)**:
   - 超过80%的智能体同时接受错误前提的比例

2. **群体置信度 (Group Confidence)**:
   - 智能体对错误答案的平均置信度

3. **错误强化循环 (Error Reinforcement Loop)**:
   - 错误前提在代际间被强化的程度

### B. 个体层面指标
1. **从众倾向指数 (Conformity Index)**:
   - 个体与群体主流意见的一致性程度

2. **独立性得分 (Independence Score)**:
   - 个体独立识别错误的能力

### C. 系统层面指标
1. **认知同质化程度 (Cognitive Homogenization)**:
   - 群体认知多样性的减少程度

2. **系统脆弱性 (System Fragility)**:
   - 系统对错误前提的整体抵抗力

## 实验设计中的应用

### 对照组设置
1. **基线组**: 单一模型，多轮测试
2. **异构组**: 多厂商多模型混合
3. **同质组**: 同一厂商多模型

### 数据收集
1. **响应内容**: 完整对话记录
2. **置信度评分**: 模型对答案的确定性
3. **响应时间**: 决策过程的反映
4. **交互模式**: 智能体间相互影响

### 统计分析
1. **相关性分析**: 共谋指标与性能的关系
2. **回归分析**: 预测因素识别
3. **聚类分析**: 识别共谋模式
4. **时间序列分析**: 追踪共谋发展

## 预期结果

### 如果存在共谋幻觉：
- 同厂商模型表现出相似的错误模式
- 群体一致性高于随机水平
- 批判性思维随时间衰减

### 如果不存在共谋幻觉：
- 模型间错误模式相互独立
- 群体一致性接近随机水平
- 批判性思维保持稳定

## 验证方法

### 交叉验证
1. **任务交叉验证**: 使用不同类型的错误前提
2. **模型交叉验证**: 在不同模型组合中测试
3. **时间交叉验证**: 在不同时间点重复测试

### 敏感性分析
1. **参数敏感性**: 测试不同参数设置的影响
2. **样本敏感性**: 测试不同样本大小的效果
3. **方法敏感性**: 比较不同测量方法的结果

这个测量方案为CHE实验提供了科学的共谋幻觉检测框架。