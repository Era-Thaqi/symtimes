import json
from sage.all import var
from helpers.types import TransitionCase
from calc.symbolic_calc import compute_output_transitions, infer_transition_cases

def load_sequence(filename):
    with open(filename, "r") as f:
        return json.load(f)

def main() -> None:
    data = load_sequence("../sequence_example.json")

    # o1: (a1, a3) → o1
    primary_times = [var(t) for t in data["input_times"]]
    a1_a3_vectors = data["input_vectors"]
    a1_vals = [v[0] for v in a1_a3_vectors]
    a3_vals = [v[1] for v in a1_a3_vectors]
    input_vectors_o1 = [list(pair) for pair in zip(a1_vals, a3_vals)]
    tc_o1 = infer_transition_cases(input_vectors_o1)
    t_o1 = compute_output_transitions(primary_times, tc_o1)

    # o0: (a0, a1) → o0 (symbolic anchors for each transition)
    a0_vals = data["a0_values"]
    a1_vals_o0 = data["a1_values"]
    input_vectors_o0 = [list(pair) for pair in zip(a0_vals, a1_vals_o0)]
    tc_o0 = infer_transition_cases(input_vectors_o0)
    o0_time_vars = [var(f"t_o0_{i+1}") for i in range(len(tc_o0))]

    input_times_o0 = []
    idx_a0 = 0
    idx_a1 = 0
    prev_vec = input_vectors_o0[0]
    for i, vec in enumerate(input_vectors_o0[1:], start=1):
        changed = [vec[j] != prev_vec[j] for j in range(2)]
        if changed == [True, False]:
            anchor = var(f"t_a0_{idx_a0 + 1}")
            idx_a0 += 1
        elif changed == [False, True]:
            anchor = var(f"t_a1_{idx_a1 + 1}")
            idx_a1 += 1
        elif changed == [True, True]:
            anchor = var(f"t_a1_{idx_a1 + 1}")  # or a0, depends on your priority
            idx_a0 += 1
            idx_a1 += 1
        else:
            raise ValueError(f"Illegal non-transition at index {i}: {vec}")
        input_times_o0.append(anchor)
        prev_vec = vec
    t_o0_formulas = compute_output_transitions(input_times_o0, tc_o0)
    o0_formula_dict = {o0_time_vars[i]: t_o0_formulas[i] for i in range(len(tc_o0))}

    # o2: (a2, o1) → o2 (symbolic anchors for each transition)
    a2_vals = data["a2_values"]
    o1_vals = [1 - (a | b) for a, b in zip(a1_vals, a3_vals)]
    input_vectors_o2 = [list(pair) for pair in zip(a2_vals, o1_vals)]
    tc_o2 = infer_transition_cases(input_vectors_o2)
    o2_time_vars = [var(f"t_o2_{i+1}") for i in range(len(tc_o2))]

    input_times_o2 = []
    idx_a2 = 0
    idx_o1 = 0
    prev_vec = input_vectors_o2[0]
    for i, vec in enumerate(input_vectors_o2[1:], start=1):
        changed = [vec[j] != prev_vec[j] for j in range(2)]
        if changed == [True, False]:
            anchor = var(f"t_a2_{idx_a2 + 1}")
            idx_a2 += 1
        elif changed == [False, True]:
            anchor = var(f"t_o1_{idx_o1 + 1}")
            idx_o1 += 1
        elif changed == [True, True]:
            anchor = var(f"t_o1_{idx_o1 + 1}")
            idx_o1 += 1
            idx_a2 += 1
        else:
            raise ValueError(f"Illegal non-transition at index {i}: {vec}")
        input_times_o2.append(anchor)
        prev_vec = vec
    t_o2_formulas = compute_output_transitions(input_times_o2, tc_o2)
    o2_formula_dict = {o2_time_vars[i]: t_o2_formulas[i] for i in range(len(tc_o2))}

    # o4: (o0, o2) → o4 (use symbolic o0/o2 variables as anchors)
    o0_vals = [1 - (a | b) for a, b in zip(a0_vals, a1_vals_o0)]
    o2_vals = [1 - (a | b) for a, b in zip(a2_vals, o1_vals)]
    input_vectors_o4 = [list(pair) for pair in zip(o0_vals, o2_vals)]
    tc_o4 = infer_transition_cases(input_vectors_o4)
    o4_time_vars = [var(f"t_o4_{i+1}") for i in range(len(tc_o4))]

    input_times_o4 = []
    idx_o0 = 0
    idx_o2 = 0
    prev_vec = input_vectors_o4[0]
    for i, vec in enumerate(input_vectors_o4[1:], start=1):
        changed = [vec[j] != prev_vec[j] for j in range(2)]
        if changed == [True, False]:
            anchor = o0_time_vars[idx_o0]
            idx_o0 += 1
        elif changed == [False, True]:
            anchor = o2_time_vars[idx_o2]
            idx_o2 += 1
        elif changed == [True, True]:
            anchor = o2_time_vars[idx_o2]  # or o0_time_vars[idx_o0]
            idx_o0 += 1
            idx_o2 += 1
        else:
            raise ValueError(f"Illegal non-transition at index {i}: {vec}")
        input_times_o4.append(anchor)
        prev_vec = vec
    t_o4_formulas = compute_output_transitions(input_times_o4, tc_o4)

    # Substitute o0 and o2 formulas into o4 for full expansion
    t_o4_expanded = []
    for expr in t_o4_formulas:
        expr_expanded = expr.substitute(o0_formula_dict).substitute(o2_formula_dict)
        t_o4_expanded.append(expr_expanded)

    # Print results
    print("o0 transition times (symbolic):")
    for i, expr in enumerate(o0_time_vars, 1):
        print(f"  t_o0_{i} =", expr)
    print("o0 transition times (formulas):")
    for i, expr in enumerate(t_o0_formulas, 1):
        print(f"  t_o0_{i} =", expr)

    print("\no1 transition times:")
    for i, expr in enumerate(t_o1, 1):
        print(f"  t_o1_{i} =", expr)

    print("\no2 transition times (symbolic):")
    for i, expr in enumerate(o2_time_vars, 1):
        print(f"  t_o2_{i} =", expr)
    print("o2 transition times (formulas):")
    for i, expr in enumerate(t_o2_formulas, 1):
        print(f"  t_o2_{i} =", expr)

    print("\no4 transition times (symbolic):")
    for i, expr in enumerate(t_o4_formulas, 1):
        print(f"  t_o4_{i} =", expr)

    print("\no4 transition times (fully expanded):")
    for i, expr in enumerate(t_o4_expanded, 1):
        print(f"  t_o4_{i} =", expr)

if __name__ == "__main__":
    main()
