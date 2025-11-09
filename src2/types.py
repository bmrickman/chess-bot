from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

State = TypeVar("State")
VisibleState = TypeVar("VisibleState")
Action = TypeVar("Action")


class Environment[VisibleState, Action](Protocol):
    def get_visible_state(self) -> VisibleState: ...
    def apply_action(self, action: Action): ...
    def game_over(self) -> bool: ...


class Agent[VisibleState, Action](Protocol):
    def update_state(self, state: VisibleState): ...
    def select_action(self) -> Action: ...
