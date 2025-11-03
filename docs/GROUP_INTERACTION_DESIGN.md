# 群体交互机制设计

## 问题分析

当前CHE实验的局限性：
1. **独立响应**：每个智能体独立回答，缺乏相互影响
2. **无信息共享**：智能体之间无法看到其他人的观点
3. **缺少群体动态**：没有观点传播、强化、共识形成过程
4. **不真实的共谋**：真正的共谋需要群体互动和相互影响

## 群体交互机制设计

### 1. 多阶段讨论模式

```
阶段1: 独立思考 (Individual Thinking)
├── 每个智能体独立分析问题
├── 记录初始观点
└── 不查看他人意见

阶段2: 观点共享 (Opinion Sharing)
├── 所有智能体查看他人观点
├── 计算观点相似性
└── 识别潜在盟友

阶段3: 群体讨论 (Group Discussion)
├── 基于相似性分组讨论
├── 互相强化或质疑观点
└── 形成群体共识

阶段4: 最终决策 (Final Decision)
├── 综合群体影响
├── 调整个人观点
└── 输出最终结论
```

### 2. 交互协议设计

#### A. 观点传播矩阵
```python
class OpinionPropagationMatrix:
    """观点传播矩阵"""

    def __init__(self, agents):
        self.agents = agents
        self.influence_matrix = self._calculate_influence_matrix()

    def _calculate_influence_matrix(self):
        """计算智能体间影响矩阵"""
        # 基于模型相似性、历史表现、角色差异等
        pass

    def propagate_opinions(self, opinions):
        """传播观点"""
        # 使用影响矩阵计算观点变化
        pass
```

#### B. 群体共识算法
```python
class GroupConsensusAlgorithm:
    """群体共识算法"""

    def calculate_consensus_level(self, opinions):
        """计算共识水平"""
        # 使用文本相似度、语义分析等
        pass

    def identify_majority_opinion(self, opinions):
        """识别多数观点"""
        # 聚类分析观点分布
        pass

    def detect_conspiracy_forming(self, opinion_history):
        """检测共谋形成"""
        # 分析观点收敛速度和模式
        pass
```

### 3. 实验流程重新设计

#### 实验组A: 独立思考组 (对照组)
- 传统模式：独立回答，无交互
- 基线数据：个体批判性思维能力

#### 实验组B: 信息共享组
- 可查看他人观点，但无讨论
- 测试信息透明度的影响

#### 实验组C: 分组讨论组
- 小组内充分讨论
- 测试群体动力学的局部影响

#### 实验组D: 全体讨论组
- 全体成员充分讨论
- 测试完整群体交互的影响

### 4. 共谋幻觉检测指标

#### A. 观点收敛速度
```python
def opinion_convergence_speed(opinion_history):
    """计算观点收敛速度"""
    # 追踪观点相似性随时间的变化
    # 快速收敛可能表明从众心理
    pass
```

#### B. 多样性损失指数
```python
def diversity_loss_index(initial_opinions, final_opinions):
    """计算多样性损失指数"""
    # 比较初始和最终观点的多样性
    # 大幅损失表明群体思维
    pass
```

#### C. 共谋强化模式
```python
def conspiracy_reinforcement_pattern(discussion_records):
    """识别共谋强化模式"""
    # 分析讨论中的错误强化循环
    # 识别相互引用错误前提的模式
    pass
```

### 5. 数据结构扩展

#### A. 讨论会话记录
```python
@dataclass
class DiscussionSession:
    """讨论会话记录"""
    session_id: str
    task_id: str
    participants: List[str]
    discussion_rounds: List[DiscussionRound]
    final_consensus: str
    consensus_level: float
    conspiracy_indicators: Dict[str, float]
```

#### B. 讨论轮次记录
```python
@dataclass
class DiscussionRound:
    """讨论轮次记录"""
    round_number: int
    speaker: str
    message: str
    target_agents: List[str]  # 针对的智能体
    influence_score: float
    opinion_change: Dict[str, float]  # 对其他智能体观点的影响
```

### 6. 实现步骤

#### 阶段1: 基础架构
1. 扩展ConversationRecord支持交互记录
2. 实现观点传播矩阵
3. 添加群体共识算法

#### 阶段2: 交互机制
1. 实现多阶段讨论流程
2. 添加信息共享机制
3. 实现观点传播算法

#### 阶段3: 检测系统
1. 实现共谋幻觉检测指标
2. 添加实时监测功能
3. 开发可视化工具

#### 阶段4: 实验验证
1. 对比实验组与对照组
2. 验证检测指标有效性
3. 优化算法参数

### 7. 预期效果

通过这个重新设计，我们将能够：

1. **真实模拟群体交互**：智能体之间可以相互影响
2. **检测共谋形成过程**：实时监测观点如何收敛
3. **量化共谋风险**：使用科学指标评估风险等级
4. **验证干预效果**：测试不同干预策略的效果

这个设计将使CHE实验更加真实和有意义。