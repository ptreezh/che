# CHE原型MVP - 开发任务清单 (v2)

本文档遵循`SPEC.md` v1.0，并将所有开发任务分解。所有涉及逻辑实现的任务都必须由测试驱动。

## 第三阶段: 原型实现与验证

### P3.1: 核心组件骨架 (`src/`)
- [ ] **任务 3.1.1:** 创建 `src/che/__init__.py` 以定义包, 及 `src/che/task.py` 并实现 `Task` dataclass。 *(Links to: SPEC 3.1)*
- [ ] **任务 3.1.2:** 创建 `src/che/agent.py` 并实现 `Agent` 抽象基类。 *(Links to: SPEC 3.1)*
- [ ] **任务 3.1.3:** 创建 `src/che/evaluator.py` 并实现 `evaluate_hallucination` 函数的骨架。 *(Links to: SPEC 3.1)*
- [ ] **任务 3.1.4:** 创建 `src/che/ecosystem.py` 并实现 `Ecosystem` 类的骨架 (`__init__`)。 *(Links to: SPEC 3.1)*

### P3.2: 核心逻辑实现 (TDD)
- [ ] **任务 3.2.1 (TDD Required):** 为 `evaluate_hallucination` 函数编写测试并实现其逻辑。 *(Links to: SPEC 3.1)*
- [ ] **任务 3.2.2 (TDD Required):** 为 `Ecosystem.evolve` 方法编写测试并实现其优胜劣汰逻辑。 *(Links to: SPEC 3.2)*
- [ ] **任务 3.2.3 (TDD Required):** 为 `Ecosystem.run_generation` 方法编写测试并实现其任务分发与评估调用逻辑。 *(Links to: SPEC 3.2)*

### P3.3: 模拟智能体与主流程
- [ ] **任务 3.3.1:** 创建 `src/che/agents/mock_agent.py` 并实现一个具体的 `MockAgent` 类。 *(Links to: SPEC 3.1)*
- [ ] **任务 3.3.2:** 创建 `main.py` 脚本，集成并运行完整的演化流程。 *(Links to: SPEC 3.2)*
- [ ] **任务 3.3.3:** 在 `main.py` 中添加日志记录，清晰地打印每一代演化后的系统平均分。 *(Links to: SPEC 1.3)*
