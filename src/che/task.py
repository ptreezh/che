from dataclasses import dataclass

@dataclass
class Task:
    """A simple data structure to hold task details."""
    instruction: str
    false_premise: str
