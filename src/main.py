import json
from sage.all import var
from helpers.types import TransitionCase
from calc.symbolic_calc import compute_output_transitions, infer_transition_cases


def load_sequence(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def main():
    data = load_sequence("../sequence_example.json")

    # Dynamically define symbolic time variables
    input_times = [var(t_name) for t_name in data["input_times"]]
    input_vectors = data["input_vectors"]  # e.g. [[0,0], [0,1], ...]

    # Infer transition cases
    transition_cases = infer_transition_cases(input_vectors)

    output = compute_output_transitions(input_times, transition_cases)

    for i, expr in enumerate(output):
        print(f"t_o_{i+1} =", expr)


if __name__ == "__main__":
    main()