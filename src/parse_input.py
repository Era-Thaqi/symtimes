"""Tmp."""
import json
from typing import Any, Dict, List, TypedDict

class Gate(TypedDict):
    name: str
    type: str
    inputs: List[str]
    output: str
    input_order: List[str]

class Circuit(TypedDict):
    inputs: List[str]
    outputs: List[str]
    gates: List[Gate]
    initial_values: Dict[str, int]

def load_circuit(filepath: str) -> Circuit:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data  # You can cast this to Circuit if using a type checker

# TEST: Run this file directly to print what's inside the JSON
if __name__ == "__main__":
    print("Starting to load circuit from input.json")
    circuit = load_circuit("input.json")
    print("Inputs:", circuit["inputs"])
    print("Outputs:", circuit["outputs"])
    print("Initial values:", circuit["initial_values"])
    print("Gates:")
    for g in circuit["gates"]:
        print(f"  {g['name']}: {g['inputs']} -> {g['output']}")
