# 任务清单: 觉醒者智能体 (Awakened Agent) 功能实现

## P3-T1: 创建 `docs/AWAKENED_AGENT_TASK_LIST.md` (已完成)

## P3-T2: 编写针对“觉醒者”智能体功能的单元测试 (RED阶段)

*   **目标:** 编写失败的测试用例，以验证“觉醒者”智能体的核心行为和成功标准。
*   **子任务:**
    *   **P3-T2.1: 定义 `AWAKENED_PROMPT` 常量。**
        *   在 `main.py` 或一个合适的配置模块中，定义 `AWAKENED_PROMPT` 字符串常量，其内容为SRD中定义的“觉醒者”提示词。
    *   **P3-T2.2: 编写 `test_ollama_agent_awakened_prompt_configuration` 测试。**
        *   **文件:** `tests/test_evaluator.py` (或新建 `tests/test_agent.py`)
        *   **内容:** 实例化 `OllamaAgent` 时传入包含 `AWAKENED_PROMPT` 的配置，验证 `OllamaAgent` 内部的 `config` 属性是否正确存储了该提示词。
        *   **预期结果:** 测试失败（因为 `OllamaAgent` 尚未支持 `AWAKENED_PROMPT`）。
    *   **P3-T2.3: 编写 `test_ollama_agent_awakened_execution_behavior` 测试。**
        *   **文件:** `tests/test_evaluator.py` (或新建 `tests/test_agent.py`)
        *   **内容:** 模拟 `OllamaAgent` 使用 `AWAKENED_PROMPT` 执行一个包含虚假前提的 `Task`。由于Ollama模型调用是外部依赖，此测试可能需要模拟Ollama的响应，以验证智能体输出是否体现出质疑和反驳“常识”的特质（例如，输出中包含“质疑”、“虚假前提”、“不接受”等关键词）。
        *   **预期结果:** 测试失败（因为 `OllamaAgent` 尚未实现基于 `AWAKENED_PROMPT` 的行为）。
    *   **P3-T2.4: 编写 `test_ecosystem_awakened_agent_initialization` 测试。**
        *   **文件:** `tests/test_evaluator.py` (或新建 `tests/test_ecosystem.py`)
        *   **内容:** 验证 `setup_ecosystem` 函数在传入 `AWAKENED_AGENT_RATIO` 时，能够按比例正确初始化包含“觉醒者”智能体的种群。检查生成的智能体列表中，有多少智能体的 `config` 中包含 `AWAKENED_PROMPT`。
        *   **预期结果:** 测试失败（因为 `setup_ecosystem` 尚未修改以支持“觉醒者”智能体）。

## P3-T3: 实现 `OllamaAgent` 对 `AWAKENED_PROMPT` 的支持，并修改 `main.py` 以引入“觉醒者”智能体 (GREEN阶段)

*   **目标:** 编写最少量的代码，使 P3-T2 中的所有测试通过。
*   **子任务:**
    *   **P3-T3.1: 在 `src/che/agents/ollama_agent.py` 中，修改 `OllamaAgent` 的 `__init__` 方法。**
        *   使其能够识别 `config` 字典中是否存在 `AWAKENED_PROMPT` 键，并将其作为系统提示词使用。
    *   **P3-T3.2: 在 `main.py` 中，定义 `AWAKENED_PROMPT` 常量。**
        *   将SRD中定义的“觉醒者”提示词内容赋值给 `AWAKENED_PROMPT`。
    *   **P3-T3.3: 在 `main.py` 中，修改 `setup_ecosystem` 函数。**
        *   引入 `AWAKENED_AGENT_RATIO` 常量（例如，0.1 或 0.2）。
        *   调整智能体创建逻辑，根据 `AWAKENED_AGENT_RATIO` 的比例，创建一部分“觉醒者”智能体，并将其 `config` 中的 `prompt` 设置为 `AWAKENED_PROMPT`。
    *   **P3-T3.4: 确保 `setup_ecosystem` 在创建“觉醒者”智能体时，正确地将 `AWAKENED_PROMPT` 传递给 `OllamaAgent`。**

## P3-T4: 运行测试，确保所有测试通过。

*   **目标:** 验证所有新编写和现有测试均通过，确认功能实现正确。

## P3-T5: 对代码进行重构，优化提示词管理和智能体初始化逻辑 (REFACTOR阶段)

*   **目标:** 提高代码的可读性、可维护性和可扩展性，同时不改变其外部行为。
*   **子任务:**
    *   **P3-T5.1: 考虑将 `CRITICAL_PROMPT`, `STANDARD_PROMPT`, `AWAKENED_PROMPT` 统一管理。**
        *   例如，可以创建一个 `PromptType` 枚举或一个字典来集中管理这些提示词，避免硬编码。
    *   **P3-T5.2: 优化 `setup_ecosystem` 中的智能体创建逻辑。**
        *   使其更具可读性和可扩展性，例如，可以创建一个辅助函数来根据类型和比例创建不同种类的智能体。

## P3-T6: 更新 `README.md` 和其他相关文档，反映新功能。

*   **目标:** 确保项目文档与代码同步，清晰地描述新功能。
*   **子任务:**
    *   **P3-T6.1: 更新 `README.md`。**
        *   简要介绍“觉醒者”智能体功能及其在CHE生态系统中的作用。
    *   **P3-T6.2: 更新 `docs/SPEC.md`。**
        *   如果“觉醒者”智能体的引入对CHE原型整体规格有影响，则更新相关部分。
    *   **P3-T6.3: 更新 `docs/DEVELOPMENT.md`。**
        *   如果引入了新的开发模式或测试方法，则更新相关部分。
