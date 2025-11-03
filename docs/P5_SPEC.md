# P5: Resilient, Checkpointed Evolution Framework - SPEC

## 1. Overview

This document specifies the technical requirements for adding a stateful pause and resume capability to the evolutionary experiment framework (`enhanced_experiment_1080_samples.py`). The primary goal is to make long-running experiments resilient to interruptions and allow them to be continued from the last completed generation.

## 2. Requirements

### REQ-P5.1: State Serialization

- **Description:** After each full generation of the evolutionary loop completes, the entire state of the simulation must be serialized and saved to a file.
- **Trigger:** This action must be triggered automatically at the end of each successful generation, before the next one begins.
- **State Contents:** The state object to be saved MUST include, at a minimum:
    - `generation_number`: The integer index of the generation that just completed (0-indexed).
    - `population_state`: A list of all agent objects in the current population. Each agent object must include its `id`, `model`, `role`, `system_prompt`, and current `fitness_score`.
    - `random_state`: The internal state of Python's `random` module's generator. This is critical for ensuring that if the experiment is resumed, the subsequent random choices (e.g., for mutation) are exactly the same as they would have been in an uninterrupted run.
    - `numpy_random_state`: The internal state of NumPy's random generator (`numpy.random.get_state()`), for the same reason.
    - `historical_results`: A log of the summary statistics (average score, etc.) from all previous generations.
- **File Format:** The state must be saved in **JSON** format for human readability and broad compatibility.
- **File Naming Convention:** State files should be named to clearly indicate their generation number, e.g., `evolution_state_gen_0.json`, `evolution_state_gen_1.json`, etc.
- **Location:** State files should be saved in a dedicated directory, e.g., `results/checkpoints/`.

### REQ-P5.2: State Deserialization (Resumption)

- **Description:** The main experiment script must be able to start from a previously saved state file, bypassing the initial setup and continuing the evolutionary loop from the next generation.
- **Trigger:** This mode should be activated via a command-line argument.
    - **Argument:** `--resume-from <file_path>`
    - **Behavior:** When this argument is provided, the script will not initialize a new population. Instead, it will:
        1.  Load the specified JSON state file.
        2.  Restore the `generation_number`. The loop should start at `generation_number + 1`.
        3.  Restore the `population_state` into the `self.agents` list.
        4.  Restore the `random_state` and `numpy_random_state` to ensure reproducibility.
        5.  Restore the `historical_results`.
- **Error Handling:** If the specified file does not exist or is corrupted, the script should terminate with a clear error message.

## 3. Implementation Notes

- The `EnhancedEcosystem` class in `enhanced_experiment_1080_samples.py` will be the primary location for the `save_state` and `load_state` methods.
- The `main()` function in the same script will be modified to handle the `--resume-from` command-line argument using Python's `argparse` module.
- Special care must be taken with object serialization. Dataclasses may need to be converted to dictionaries before saving to JSON. The `asdict` utility can be used.
