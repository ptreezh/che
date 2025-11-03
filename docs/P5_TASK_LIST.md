# P5: Resilient, Checkpointed Evolution Framework - Task List

This plan follows a strict Test-Driven Development (TDD) methodology.

## P5-T1: Implement State Serialization

-   **[RED]**
    -   Create a new test file `tests/test_evolution_framework.py`.
    -   Write a test `test_save_state_creates_file`.
    -   **Test Logic:**
        1.  Initialize a minimal `EnhancedEcosystem`.
        2.  Run it for exactly one generation.
        3.  Call a new (not-yet-existing) method `ecosystem.save_state()`.
        4.  Assert that a state file (e.g., `evolution_state_gen_0.json`) is created in the designated checkpoints directory.
        5.  Assert that the file content is valid JSON and not empty.
    -   **Expected Outcome:** The test fails because `save_state` does not exist.

-   **[GREEN]**
    -   Implement the `save_state(filepath)` method in the `EnhancedEcosystem` class.
    -   The method should gather the required state variables (generation, population, random states) and write them to the specified JSON file.
    -   Modify the main evolution loop in `run_evolution_experiment` to call `save_state` at the end of each generation.
    -   **Expected Outcome:** The `test_save_state_creates_file` test now passes.

-   **[REFACTOR]**
    -   Review the `save_state` implementation for clarity and efficiency.
    -   Ensure file paths are handled robustly using `os.path.join`.
    -   Add logging to indicate that a checkpoint has been saved.

## P5-T2: Implement State Deserialization (Resume)

-   **[RED]**
    -   In `tests/test_evolution_framework.py`, write a test `test_resume_from_state_loads_correctly`.
    -   **Test Logic:**
        1.  Create an ecosystem, run it for one generation, and save its state to `state_gen_0.json`. Capture the state of the agents.
        2.  Create a *new*, empty `EnhancedEcosystem` instance.
        3.  Call a new (not-yet-existing) method `new_ecosystem.load_state('state_gen_0.json')`.
        4.  Assert that `new_ecosystem.generation` is now `0`.
        5.  Assert that the agent population in `new_ecosystem.agents` is identical in structure and fitness scores to the saved state.
        6.  Assert that the random states are correctly restored.
    -   **Expected Outcome:** The test fails because `load_state` does not exist.

-   **[GREEN]**
    -   Implement the `load_state(filepath)` method in the `EnhancedEcosystem` class.
    -   The method should read the JSON file, parse it, and restore all the required state variables to the class instance (`self.generation`, `self.agents`, etc.).
    -   **Expected Outcome:** The `test_resume_from_state_loads_correctly` test now passes.

-   **[REFACTOR]**
    -   Add robust error handling for file-not-found or corrupted JSON scenarios.
    -   Refine the object reconstruction logic.

## P5-T3: Implement Command-Line Integration & Full Cycle Test

-   **[RED]**
    -   In `tests/test_evolution_framework.py`, write an integration test `test_full_run_pause_resume_cycle`.
    -   **Test Logic:**
        1.  Use `subprocess` to run the main script (`enhanced_experiment_1080_samples.py`) for 2 generations.
        2.  Assert that `evolution_state_gen_1.json` is created.
        3.  Use `subprocess` again to run the main script with a new argument: `--resume-from evolution_state_gen_1.json` for 2 more generations.
        4.  Check the script's output logs to confirm it started from "Generation 3".
        5.  Assert that `evolution_state_gen_3.json` is created.
    -   **Expected Outcome:** The test fails because the main script does not recognize the `--resume-from` argument.

-   **[GREEN]**
    -   Modify the `main()` function in `enhanced_experiment_1080_samples.py`.
    -   Use Python's `argparse` module to add and parse the `--resume-from` command-line argument.
    -   If the argument is present, call the `load_state` method before starting the evolution loop.
    -   Adjust the evolution loop's `range()` to start from the loaded generation number + 1.
    -   **Expected Outcome:** The `test_full_run_pause_resume_cycle` integration test now passes.

-   **[REFACTOR]**
    -   Clean up the `main` function logic.
    -   Ensure clear logging messages for both normal start and resumed start.
