"""Tmp."""

from enum import Enum, auto
from typing import NamedTuple, Self

from sage.symbolic.expression import Expression


class TransitionCase(Enum):
    """Tmp."""

    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()
    H = auto()


class TransitionTuple(NamedTuple):
    """Tmp."""

    time: Expression
    transition: TransitionCase


class TransitionsIterator:
    """Tmp."""

    index = 0
    transition_list: list[TransitionTuple]

    def __init__(
        self,
        transition_list: list[TransitionTuple],
    ) -> None:
        """Tmp."""
        self.transition_list = transition_list

    def __iter__(self) -> Self:
        """Tmp."""
        return self

    def __next__(self) -> tuple[TransitionCase, TransitionCase]:
        """Tmp."""
        self.index += 1
        if self.index > len(self.transition_list) - 1:
            raise StopIteration

        return (
            self.transition_list[self.index - 1].transition,
            self.transition_list[self.index].transition,
        )
