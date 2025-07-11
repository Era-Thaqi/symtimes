"""Tmp."""

from helpers.types import TransitionCase

transitions_map = {
    "A": {
        (0, 0): (1, TransitionCase.A),
        (0, 1): (1, TransitionCase.D),
        (1, 0): (0, TransitionCase.H),
        (1, 1): (0, TransitionCase.E),
    },
    "B": {
        (0, 0): (1, TransitionCase.B),
        (0, 1): (0, TransitionCase.G),
        (1, 0): (1, TransitionCase.C),
        (1, 1): (0, TransitionCase.F),
    },
}

transition_cases_map = {
    TransitionCase.A: {"from": (0, 0), "to": (1, 0), "falling": "0"},
    TransitionCase.B: {"from": (0, 0), "to": (0, 1), "falling": "0"},
    TransitionCase.C: {"from": (1, 0), "to": (1, 1), "falling": "B"},
    TransitionCase.D: {"from": (0, 1), "to": (1, 1), "falling": "0"},
    TransitionCase.E: {"from": (1, 1), "to": (0, 1), "falling": "A"},
    TransitionCase.F: {"from": (1, 1), "to": (1, 0), "falling": "B"},
    TransitionCase.G: {"from": (0, 1), "to": (0, 0), "falling": "B"},
    TransitionCase.H: {"from": (1, 0), "to": (0, 0), "falling": "A"},
}
