import os
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

import numpy as np
from numpy.typing import NDArray


StateType = TypeVar("StateType")
ObsType = TypeVar("ObsType")



class WorldCore(Generic[StateType, ObsType]):
    # state
    STATE_NAME: list[str]
    STATE_UPPER_BOUND: list[SupportsFloat]
    STATE_LOWER_BOUND: list[SupportsFloat]
    STATE_DIM: int
    # observation
    OBSERVATION_NAME: list[str]
    OBSERVATION_UPPER_BOUND: list[SupportsFloat]
    OBSERVATION_LOWER_BOUND: list[SupportsFloat]
    OBSERVATION_SCALE: list[SupportsFloat]
    OBSERVATION_DIM: int

    # obstacles
    obstacle_polygons: list[NDArray]

    def reset(self, state: StateType) -> StateType:
        raise NotImplementedError

    def step(self, dt: SupportsFloat, np_random=None) -> StateType:
        raise NotImplementedError

    def get_state(self) -> StateType:
        raise NotImplementedError

    def observe_state(self, state: StateType, np_random=None) -> ObsType:
        raise NotImplementedError

    def check_collision(self, ship_poly: NDArray) -> bool:
        if not hasattr(self, "obstacle_segments"):
            self.obstacle_segments = ploygons2segments(self.obstacle_polygons)
        ship_segments = ploygons2segments([ship_poly])
        #
        if len(self.obstacle_segments) == 0:
            return False
        else:
            for ship_segment in ship_segments:
                interect_vec = check_interect(ship_segment, self.obstacle_segments)
                interect = bool(np.any(interect_vec))
                if interect:
                    return True
        return False


def ploygons2segments(polygons: list[NDArray]) -> list[NDArray]:
    segments = []
    for polygon in polygons:
        segment_num = len(polygon)
        for i in range(segment_num):
            segment = polygon[[i % segment_num, (i + 1) % segment_num], :]
            segments.append(segment)
    return segments


def check_interect(base_segment: NDArray, segments: list[NDArray]) -> bool:
    p1 = base_segment[0, :]
    p2 = base_segment[1, :]
    p3 = np.array(segments)[:, 0, :]
    p4 = np.array(segments)[:, 1, :]
    #
    t1 = (p1[0] - p2[0]) * (p3[:, 1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[:, 0])
    t2 = (p1[0] - p2[0]) * (p4[:, 1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[:, 0])
    t3 = (p3[:, 0] - p4[:, 0]) * (p1[1] - p3[:, 1]) + (p3[:, 1] - p4[:, 1]) * (
        p3[:, 0] - p1[0]
    )
    t4 = (p3[:, 0] - p4[:, 0]) * (p2[1] - p3[:, 1]) + (p3[:, 1] - p4[:, 1]) * (
        p3[:, 0] - p2[0]
    )
    return (t1 * t2 < 0) & (t3 * t4 < 0)
