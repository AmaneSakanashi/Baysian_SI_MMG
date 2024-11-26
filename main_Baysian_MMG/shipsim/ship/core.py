import os
from typing import TYPE_CHECKING, Generic, SupportsFloat, TypeVar, List

import numpy as np
from numpy.typing import NDArray


StateType = TypeVar("StateType")
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class ShipCore(Generic[StateType, ActType, ObsType]):
    # principle particular
    L: float
    B: float
    # time
    TIME_NAME: list[str] = ["t [s]"]
    # state
    STATE_NAME: list[str]
    STATE_UPPER_BOUND: list[SupportsFloat]
    STATE_LOWER_BOUND: list[SupportsFloat]
    STATE_DIM: int
    # action
    ACTION_NAME: list[str]
    ACTION_UPPER_BOUND: list[SupportsFloat]
    ACTION_LOWER_BOUND: list[SupportsFloat]
    ACTION_DIM: int
    # observation
    OBSERVATION_NAME: list[str]
    OBSERVATION_UPPER_BOUND: list[SupportsFloat]
    OBSERVATION_LOWER_BOUND: list[SupportsFloat]
    OBSERVATION_SCALE: list[SupportsFloat]
    OBSERVATION_DIM: int

    def ode_rhs(self, state: StateType, action: ActType) -> StateType:
        raise NotImplementedError

    def observe_state(self, state: StateType, np_random=None) -> ObsType:
        raise NotImplementedError

    def ship_polygon(self, state: StateType) -> NDArray:
        raise NotImplementedError
