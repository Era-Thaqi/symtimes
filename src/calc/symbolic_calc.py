from sage.all import var, assume, exp, ln, sqrt, Integer
from sage.symbolic.expression import Expression
from helpers.types import TransitionCase

# ---- SYMBOLIC VARIABLES ----
A = var("A"); A_OVERLINE = var("A_overline", latex_name="\\overline{A}")
SMALL_A = var("a")
ALPHA1 = var("alpha_1"); ALHPA2 = var("alpha_2")
C1 = var("C_1"); C1_PRIME = var("C1_prime")
C2 = var("C_2"); C3 = var("C_3")
SMALL_CHI = var("chi"); SMALL_CHI_OVERLINE = var("chi_overline", latex_name="\\overline{\\chi}")
DELTA_MIN = var("delta_min")
R = var("R")
RnA = var("R_nA"); RnB = var("R_nB")
W_1 = var("W_1")

# Assumptions
for sym in [A, A_OVERLINE, SMALL_A, ALPHA1, ALHPA2, C1, C1_PRIME, C2, C3, SMALL_CHI, SMALL_CHI_OVERLINE, DELTA_MIN, R, RnA, RnB, W_1]:
    assume(sym > 0)

# ---- INITIAL TRANSITIONS A–H ----
def transition_a(_, __, ___): return C1 * RnA * ln(2) + DELTA_MIN
def transition_b(_, __, ___): return C1_PRIME * RnB * ln(2) + DELTA_MIN
def transition_c(_, __, ___): return C1 * RnA * ln(2) + DELTA_MIN
def transition_d(_, __, ___): return C1_PRIME * RnB * ln(2) + DELTA_MIN
def transition_e(_, __, ___): return C1_PRIME * RnB * ln(2) + DELTA_MIN
def transition_f(_, __, ___): return C1 * RnA * ln(2) + DELTA_MIN
def transition_g(_, __, ___): return C1_PRIME * RnB * ln(2) + DELTA_MIN
def transition_h(_, __, ___): return C1 * RnA * ln(2) + DELTA_MIN

def infer_transition_cases(input_vectors):
    """
    Infers transition cases for a sequence of input vectors.
    The first case is based on the initial two vectors.
    The remaining cases are based on sliding triplets.
    """
    # --- Initial two-vector transition logic ---
    def transition_case_two(vec1, vec2):
        if vec1 == [0, 0] and vec2 == [1, 0]: return TransitionCase.A
        if vec1 == [0, 0] and vec2 == [0, 1]: return TransitionCase.B
        if vec1 == [1, 0] and vec2 == [1, 1]: return TransitionCase.C
        if vec1 == [0, 1] and vec2 == [1, 1]: return TransitionCase.D
        if vec1 == [1, 1] and vec2 == [0, 1]: return TransitionCase.E
        if vec1 == [1, 1] and vec2 == [1, 0]: return TransitionCase.F
        if vec1 == [0, 1] and vec2 == [0, 0]: return TransitionCase.G
        if vec1 == [1, 0] and vec2 == [0, 0]: return TransitionCase.H
        raise ValueError(f"No initial transition mapping for {vec1} → {vec2}")

    # --- Triplet transition logic ---
    def triplet_to_case(prev, curr, next_):
        if prev == [1, 0] and curr == [0, 0] and next_ == [1, 0]: return TransitionCase.A  # (h,a)
        if prev == [0, 1] and curr == [0, 0] and next_ == [1, 0]: return TransitionCase.A  # (g,a)
        if prev == [1, 0] and curr == [0, 0] and next_ == [0, 1]: return TransitionCase.B  # (h,b)
        if prev == [0, 1] and curr == [0, 0] and next_ == [0, 1]: return TransitionCase.B  # (g,b)
        if prev == [0, 0] and curr == [1, 0] and next_ == [1, 1]: return TransitionCase.C  # (a,c)
        if prev == [1, 1] and curr == [1, 0] and next_ == [1, 1]: return TransitionCase.C  # (f,c)
        if prev == [0, 0] and curr == [0, 1] and next_ == [1, 1]: return TransitionCase.D  # (b,d)
        if prev == [1, 1] and curr == [0, 1] and next_ == [1, 1]: return TransitionCase.D  # (e,d)
        if prev == [1, 0] and curr == [1, 1] and next_ == [0, 1]: return TransitionCase.E  # (c,e)
        if prev == [0, 1] and curr == [1, 1] and next_ == [0, 1]: return TransitionCase.E  # (d,e)
        if prev == [1, 0] and curr == [1, 1] and next_ == [1, 0]: return TransitionCase.F  # (c,f)
        if prev == [0, 1] and curr == [1, 1] and next_ == [1, 0]: return TransitionCase.F  # (d,f)
        if prev == [0, 0] and curr == [0, 1] and next_ == [0, 0]: return TransitionCase.G  # (b,g)
        if prev == [1, 1] and curr == [0, 1] and next_ == [0, 0]: return TransitionCase.G  # (e,g)
        if prev == [0, 0] and curr == [1, 0] and next_ == [0, 0]: return TransitionCase.H  # (a,h)
        if prev == [1, 1] and curr == [1, 0] and next_ == [0, 0]: return TransitionCase.H  # (f,h)
        raise ValueError(f"No transition mapping for {prev} → {curr} → {next_}")

    cases = []
    # Initial transition (first two vectors)
    cases.append(transition_case_two(input_vectors[0], input_vectors[1]))
    # Remaining transitions (triplets)
    for i in range(0, len(input_vectors) - 2):
        cases.append(triplet_to_case(input_vectors[i], input_vectors[i+1], input_vectors[i+2]))
    return cases
# Map TransitionCase to initial function
t0_case_map = {
    TransitionCase.A: transition_a,
    TransitionCase.B: transition_b,
    TransitionCase.C: transition_c,
    TransitionCase.D: transition_d,
    TransitionCase.E: transition_e,
    TransitionCase.F: transition_f,
    TransitionCase.G: transition_g,
    TransitionCase.H: transition_h,
}
def transition_ha(
    delta: Expression,
    delta_prime: Expression,
    large_t: Expression,
) -> Expression:
    """Tmp."""
    start_term = C1 * RnA
    exponential_term = exp(-(large_t + DELTA_MIN) / (Integer("2") * R * C3))

    first_exponent = (-A_OVERLINE + SMALL_A) / (Integer("2") * R * C3)
    second_exponent = (A_OVERLINE) / (Integer("2") * R * C3)

    if delta_prime >= large_t + DELTA_MIN:
        first_bracket = Integer("1") + (
            (Integer("2") * (large_t + DELTA_MIN))
            / (
                SMALL_A
                + abs(delta)
                + sqrt(SMALL_CHI_OVERLINE)
                + Integer("2") * (delta_prime - large_t - DELTA_MIN)
            )
        )

        second_bracket = Integer("1") + (
            (Integer("2") * (large_t + DELTA_MIN))
            / (
                SMALL_A
                + abs(delta)
                - sqrt(SMALL_CHI_OVERLINE)
                + Integer("2") * (delta_prime - large_t - DELTA_MIN)
            )
        )
    else:
        first_bracket = Integer("1") + (
            (Integer("2") * delta_prime)
            / (SMALL_A + abs(delta) + sqrt(SMALL_CHI_OVERLINE))
        )

        second_bracket = Integer("1") + (
            (Integer("2") * delta_prime)
            / (SMALL_A + abs(delta) - sqrt(SMALL_CHI_OVERLINE))
        )

    return start_term * ln(
        Integer("2")
        - exponential_term
        * (first_bracket**first_exponent)
        * (second_bracket**second_exponent),
    )


def transition_ga(
    delta: Expression,
    delta_prime: Expression,
    large_t: Expression,
) -> Expression:
    """Tmp."""
    start_term = C1 * RnA
    exponential_term = exp(-(large_t + DELTA_MIN) / (Integer("2") * R * C3))

    first_exponent = (-A_OVERLINE + SMALL_A) / (Integer("2") * R * C3)
    second_exponent = (A_OVERLINE) / (Integer("2") * R * C3)

    if delta_prime >= large_t + DELTA_MIN:
        first_bracket = Integer("1") + (
            (Integer("2") * (large_t + DELTA_MIN))
            / (
                SMALL_A
                + delta
                + sqrt(SMALL_CHI_OVERLINE)
                + Integer("2") * (delta_prime - large_t - DELTA_MIN)
            )
        )

        second_bracket = Integer("1") + (
            (Integer("2") * (large_t + DELTA_MIN))
            / (
                SMALL_A
                + delta
                - sqrt(SMALL_CHI_OVERLINE)
                + Integer("2") * (delta_prime - large_t - DELTA_MIN)
            )
        )
    else:
        first_bracket = Integer("1") + (
            (Integer("2") * delta_prime) / (SMALL_A + delta + sqrt(SMALL_CHI_OVERLINE))
        )

        second_bracket = Integer("1") + (
            (Integer("2") * delta_prime) / (SMALL_A + delta - sqrt(SMALL_CHI_OVERLINE))
        )

    return start_term * ln(
        Integer("2")
        - exponential_term
        * (first_bracket**first_exponent)
        * (second_bracket**second_exponent),
    )


def transition_hb(
    delta: Expression,
    delta_prime: Expression,
    large_t: Expression,
) -> Expression:
    """Tmp."""
    start_term = C1_PRIME * RnB
    exponential_term = exp(-(large_t + DELTA_MIN) / (Integer("2") * R * C3))

    first_exponent = (-A_OVERLINE + SMALL_A) / (Integer("2") * R * C3)
    second_exponent = (A_OVERLINE) / (Integer("2") * R * C3)

    if delta_prime >= large_t + DELTA_MIN:
        first_bracket = Integer("1") + (
            (Integer("2") * (large_t + DELTA_MIN))
            / (
                SMALL_A
                + abs(delta)
                + sqrt(SMALL_CHI_OVERLINE)
                + Integer("2") * (delta_prime - large_t - DELTA_MIN)
            )
        )

        second_bracket = Integer("1") + (
            (Integer("2") * (large_t + DELTA_MIN))
            / (
                SMALL_A
                + abs(delta)
                - sqrt(SMALL_CHI_OVERLINE)
                + Integer("2") * (delta_prime - large_t - DELTA_MIN)
            )
        )
    else:
        first_bracket = Integer("1") + (
            (Integer("2") * delta_prime)
            / (SMALL_A + abs(delta) + sqrt(SMALL_CHI_OVERLINE))
        )

        second_bracket = Integer("1") + (
            (Integer("2") * delta_prime)
            / (SMALL_A + abs(delta) - sqrt(SMALL_CHI_OVERLINE))
        )

    return start_term * ln(
        Integer("2")
        - exponential_term
        * (first_bracket**first_exponent)
        * (second_bracket**second_exponent),
    )


def transition_gb(
    delta: Expression,
    delta_prime: Expression,
    large_t: Expression,
) -> Expression:
    """Tmp."""
    start_term = C1_PRIME * RnB
    exponential_term = exp(-(large_t + DELTA_MIN) / (Integer("2") * R * C3))

    first_exponent = (-A_OVERLINE + SMALL_A) / (Integer("2") * R * C3)
    second_exponent = (A_OVERLINE) / (Integer("2") * R * C3)

    if delta_prime >= large_t + DELTA_MIN:
        first_bracket = Integer("1") + (
            (Integer("2") * (large_t + DELTA_MIN))
            / (
                SMALL_A
                + delta
                + sqrt(SMALL_CHI_OVERLINE)
                + Integer("2") * (delta_prime - large_t - DELTA_MIN)
            )
        )

        second_bracket = Integer("1") + (
            (Integer("2") * (large_t + DELTA_MIN))
            / (
                SMALL_A
                + delta
                - sqrt(SMALL_CHI_OVERLINE)
                + Integer("2") * (delta_prime - large_t - DELTA_MIN)
            )
        )
    else:
        first_bracket = Integer("1") + (
            (Integer("2") * delta_prime) / (SMALL_A + delta + sqrt(SMALL_CHI_OVERLINE))
        )

        second_bracket = Integer("1") + (
            (Integer("2") * delta_prime) / (SMALL_A + delta - sqrt(SMALL_CHI_OVERLINE))
        )

    return start_term * ln(
        Integer("2")
        - exponential_term
        * (first_bracket**first_exponent)
        * (second_bracket**second_exponent),
    )


def transition_ac_fc(
    _delta: Expression,
    _delta_prime: Expression,
    large_t: Expression,
) -> Expression:
    """Tmp."""
    return ((-C2 * RnB * (large_t + DELTA_MIN)) / (C1 * (RnA + RnB))) + DELTA_MIN


def transition_bd_ed(
    _delta: Expression,
    _delta_prime: Expression,
    large_t: Expression,
) -> Expression:
    """Tmp."""
    return ((-C2 * RnA * (large_t + DELTA_MIN)) / (C1_PRIME * (RnA + RnB))) + DELTA_MIN


def transition_ce_de(
    _delta: Expression,
    _delta_prime: Expression,
    large_t: Expression,
) -> Expression:
    """Tmp."""
    return ((-C1_PRIME * (RnA + RnB) * (large_t + DELTA_MIN)) / (C2 * RnA)) + DELTA_MIN


def transition_cf_df(
    _delta: Expression,
    _delta_prime: Expression,
    large_t: Expression,
) -> Expression:
    """Tmp."""
    return ((-C1 * (RnA + RnB) * (large_t + DELTA_MIN)) / (C2 * RnB)) + DELTA_MIN


def transition_bg_eg(
    delta: Expression,
    _delta_prime: Expression,
    large_t: Expression,
) -> Expression:
    """Tmp."""
    if large_t + DELTA_MIN >= 0:
        if delta >= 0 and delta < (
            ((ALPHA1 + ALHPA2) * delta_0(large_t) - delta_oo(large_t)) / (ALPHA1)
        ):
            return delta_0(large_t) - ((ALPHA1) / (ALPHA1 + ALHPA2)) * delta + DELTA_MIN
        return delta_oo(large_t) + DELTA_MIN
    return (
        -Integer("2")
        * R
        * C3
        * ln(
            (Integer("1"))
            / (Integer("2") - exp((-large_t - DELTA_MIN) / (C1_PRIME * RnB))),
        )
        + DELTA_MIN
    )


def transition_ah_fh(
    delta: Expression,
    _delta_prime: Expression,
    large_t: Expression,
) -> Expression:
    """Tmp."""
    if large_t + DELTA_MIN >= 0:
        if delta >= 0 and delta < (
            ((ALPHA1 + ALHPA2) * delta_minus_0(large_t) - delta_minus_oo(large_t))
            / (ALPHA1)
        ):
            return (
                delta_minus_0(large_t)
                - ((ALHPA2) / (ALPHA1 + ALHPA2)) * abs(delta)
                + DELTA_MIN
            )
        return delta_minus_oo(large_t) + DELTA_MIN
    return (
        -Integer("2")
        * R
        * C3
        * ln(
            (Integer("1")) / (Integer("2") - exp((-large_t - DELTA_MIN) / (C1 * RnA))),
        )
        + DELTA_MIN
    )


def delta_0(large_t: Expression) -> Expression:
    """Tmp."""
    return -((ALPHA1 + ALHPA2) / (Integer("2") * R)) * (
        Integer("1")
        + W_1
        * (
            (-Integer("1"))
            / (
                e
                * (Integer("2") - exp((-large_t - DELTA_MIN) / (C1_PRIME * RnB)))
                ** ((Integer("4") * R ** Integer("2") * C3) / (ALPHA1 + ALHPA2))
            )
        )
    )


def delta_oo(large_t: Expression) -> Expression:
    """Tmp."""
    return -((ALHPA2) / (Integer("2") * R)) * (
        Integer("1")
        + W_1
        * (
            (-Integer("1"))
            / (
                e
                * (Integer("2") - exp((-large_t - DELTA_MIN) / (C1_PRIME * RnB)))
                ** ((Integer("4") * R ** Integer("2") * C3) / (ALHPA2))
            )
        )
    )


def delta_minus_0(large_t: Expression) -> Expression:
    """Tmp."""
    return -((ALPHA1 + ALHPA2) / (Integer("2") * R)) * (
        Integer("1")
        + W_1
        * (
            (-Integer("1"))
            / (
                e
                * (Integer("2") - exp((-large_t - DELTA_MIN) / (C1 * RnB)))
                ** ((Integer("4") * R ** Integer("2") * C3) / (ALPHA1 + ALHPA2))
            )
        )
    )


def delta_minus_oo(large_t: Expression) -> Expression:
    """Tmp."""
    return -((ALPHA1) / (Integer("2") * R)) * (
        Integer("1")
        + W_1
        * (
            (-Integer("1"))
            / (
                e
                * (Integer("2") - exp((-large_t - DELTA_MIN) / (C1 * RnB)))
                ** ((Integer("4") * R ** Integer("2") * C3) / (ALPHA1))
            )
        )
    )


# ---- TRANSITION MAPPING ----
sequence_map = {
    (TransitionCase.H, TransitionCase.A): transition_ha,
    (TransitionCase.G, TransitionCase.A): transition_ga,
    (TransitionCase.H, TransitionCase.B): transition_hb,
    (TransitionCase.G, TransitionCase.B): transition_gb,
    (TransitionCase.A, TransitionCase.C): transition_ac_fc,
    (TransitionCase.F, TransitionCase.C): transition_ac_fc,
    (TransitionCase.B, TransitionCase.D): transition_bd_ed,
    (TransitionCase.E, TransitionCase.D): transition_bd_ed,
    (TransitionCase.C, TransitionCase.E): transition_ce_de,
    (TransitionCase.D, TransitionCase.E): transition_ce_de,
    (TransitionCase.C, TransitionCase.F): transition_cf_df,
    (TransitionCase.D, TransitionCase.F): transition_cf_df,
    (TransitionCase.B, TransitionCase.G): transition_bg_eg,
    (TransitionCase.E, TransitionCase.G): transition_bg_eg,
    (TransitionCase.A, TransitionCase.H): transition_ah_fh,
    (TransitionCase.F, TransitionCase.H): transition_ah_fh,
    # Add the rest: (TransitionCase.G, TransitionCase.A): transition_ga, etc.
}

# ---- MAIN COMPUTATION ENGINE ----
def get_t0_formula(case):
    if case not in t0_case_map:
        raise NotImplementedError(f"Initial case {case} not implemented")
    return t0_case_map[case](None, None, None)

def get_transition_formula(prev_case, curr_case):
    if (prev_case, curr_case) not in sequence_map:
        raise NotImplementedError(f"Transition ({prev_case}, {curr_case}) not implemented")
    return sequence_map[(prev_case, curr_case)]

def compute_output_transitions(input_times, transition_cases):
    t_out = []

    for i, curr_case in enumerate(transition_cases):
        if i == 0:
            t_o = input_times[i] + get_t0_formula(curr_case)
        else:
            prev_case = transition_cases[i - 1]
            T = input_times[i] - t_out[i - 1]
            delta = t_out[i - 1] - input_times[i - 1]
            delta_prime = t_out[i - 2] - input_times[i - 2] if i >= 2 else delta
            formula = get_transition_formula(prev_case, curr_case)
            t_o = input_times[i] + formula(delta, delta_prime, T)

        t_out.append(t_o)

    return t_out
