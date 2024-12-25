"""
utils
"""

from typing import Any
from logging import Logger, getLogger
import sys

import numpy as np
from numpy.linalg import norm

from dynamics.body.body_base import BodyBase
from dynamics.body.bodies import Bodies
from dynamics.body.fixed_body_base import FixedBodyBase
from dynamics.force.force_base import ForceBase
from dynamics.force.forces import Forces
from dynamics.instant_creators.constants import Constants
from dynamics.instant_creators.rigid_ball_creator import RigidBallCreator
from dynamics.instant_creators.vertical_wall_1d_creator import VerticalWall1DCreator
from dynamics.instant_creators.vertical_pin_2d_creator import VerticalPin2DCreator
from dynamics.instant_creators.spring_creator import SpringCreator
from dynamics.instant_creators.gtavity_like_creator import GravityLikeCreator
from dynamics.instant_creators.frictional_force_2d_creator import FrictionalForce2DCreator
from dynamics.instant_creators.non_sticky_left_horizontal_spring_creator import (
    NonStickyLeftHorizontalSpringCreator,
)
from dynamics.instant_creators.horizontal_frictional_force_1d_creator import (
    HorizontalFrictionalForce1DCreator,
)

logger: Logger = getLogger()

_SQUARE_X_COORDINATES: np.ndarray = np.array([0, 1, 1, 0], float)
_SQUARE_Y_COORDINAETS: np.ndarray = np.array([0, 0, 1, 1], float)


def energy_info(
    bodies: Bodies, forces: Forces
) -> tuple[list[str], np.ndarray, tuple[np.ndarray, ...]]:
    ke: float = bodies.kinetic_energy
    bpe: float = bodies.potential_energy(forces)
    fpe: float = forces.potential_energy
    pe: float = bpe + fpe
    de: float = bodies.dissipated_energy

    force_potential_energy_bar_vertices: np.ndarray = np.vstack(
        (_SQUARE_X_COORDINATES, bpe + fpe * _SQUARE_Y_COORDINAETS)
    )
    kinetic_energy_bar_vertices: np.ndarray = np.vstack(
        (_SQUARE_X_COORDINATES, (bpe + fpe) + ke * _SQUARE_Y_COORDINAETS)
    )
    dissipated_energy_bar_vertices: np.ndarray = np.vstack(
        (_SQUARE_X_COORDINATES, (bpe + fpe + ke) + de * _SQUARE_Y_COORDINAETS)
    )

    return (
        [
            r"$E_\mathrm{k} + E_\mathrm{p} + E_\mathrm{d}$ = "
            + f"{ke+pe+de:.2f}"
            + r", $E_\mathrm{k} + E_\mathrm{p}$ = "
            + f"{ke+pe:.2f}",
            r"$E_\mathrm{k}$ = "
            + f"{ke:.2f}, "
            + r"$E_\mathrm{p}$ = "
            + f"{pe:.2f}"
            + r" (= $E_\mathrm{p,gravity}$ "
            + f"({bpe:.2f})"
            + r" + $E_\mathrm{p,spring}$ "
            + f"({fpe:.2f})"
            + r"), $E_\mathrm{d}$ = "
            + f"{de:.2f}",
        ],
        np.array([ke, bpe, fpe, de], float),
        (
            force_potential_energy_bar_vertices,
            kinetic_energy_bar_vertices,
            dissipated_energy_bar_vertices,
        ),
    )


def kinematics_info_text(bodies: Bodies) -> list[str]:
    return [
        "$l$ = ("
        + ", ".join([f"{x:.2f}" for x in body.loc])
        + "), $v$ = ("
        + ", ".join([f"{x:.2f}" for x in body.vel])
        + f"), $\|v\|$ = {norm(body.vel):.2f}"  # noqa:W605
        + r"& $E_\mathrm{d}$ = "
        + f"{body.dissipated_energy:.2f}"
        for body in bodies.bodies
        if not isinstance(body, FixedBodyBase)
    ]


def load_dynamic_system_simulation_setting(
    data: dict[str, Any]
) -> tuple[dict[str, str | float | int | bool | list[int | float]], Bodies, Forces]:
    # data hierarchy check
    assert "name" in data
    # assert "constants" in data
    assert "rigid_ball" in data

    for key, value in data.items():
        if key == "name":
            assert isinstance(value, str), value.__class__
        elif key == "constants" or key == "simulation_setting":
            assert isinstance(value, dict), value.__class__
        else:
            assert isinstance(value, list), (value, value.__class__)
            for v in value:
                assert isinstance(v, dict), v.__class__

    _data: dict[str, Any] = data.copy()

    # parse constants
    constants: Constants = Constants()

    if "constants" in _data:
        for key, value in _data.pop("constants").items():
            constants.assign(key, value)

    # parse simulation setting

    simulation_setting: dict[str, str | float | int | bool | list[int | float]] = (
        _data.pop("simulation_setting").copy()
        if "simulation_setting" in _data
        else dict(minimize_energy=False)
    )
    simulation_setting["minimize_energy"] = simulation_setting.get("minimize_energy", False)
    assert "name" not in simulation_setting
    simulation_setting["name"] = _data.pop("name")
    simulation_setting["grid"] = simulation_setting.get("grid", False)
    simulation_setting["1d"] = simulation_setting.get("1d", False)
    simulation_setting["energy_bar_padding"] = simulation_setting.get("energy_bar_padding", 0.6)
    simulation_setting["save_to_gif"] = simulation_setting.get("save_to_gif", False)

    if "sim_time_step" in simulation_setting:
        simulation_setting["sim_time_step"] = constants.value(
            simulation_setting["sim_time_step"]  # type:ignore
        )
    if "sim_time_step_const_vel" in simulation_setting:
        simulation_setting["sim_time_step_const_vel"] = constants.value(
            simulation_setting["sim_time_step_const_vel"]  # type:ignore
        )

    if simulation_setting["save_to_gif"]:
        simulation_setting["gif_filepath"] = simulation_setting.get(
            "gif_filepath", "dynamics-simulation.gif"
        )
        simulation_setting["num_frames_per_sec"] = simulation_setting.get(
            "num_frames_per_sec",
            int(1.0 / simulation_setting["real_world_time_interval"]),  # type:ignore
        )
        simulation_setting["num_frames_saved"] = simulation_setting.get("num_frames_saved", 250)

    # parse body information

    id_body_map: dict[str, BodyBase] = dict()

    for rigid_ball_data in _data.pop("rigid_ball"):
        _id, rigid_ball = RigidBallCreator.create(rigid_ball_data, constants)
        assert _id not in id_body_map, (list(id_body_map.keys()), _id)
        id_body_map[_id] = rigid_ball

    if "vertical_pin_2d" in _data:
        for vertical_pin_2d_data in _data.pop("vertical_pin_2d"):
            _id, vertical_pin_2d = VerticalPin2DCreator.create(vertical_pin_2d_data)
            assert _id not in id_body_map, (list(id_body_map.keys()), _id)
            id_body_map[_id] = vertical_pin_2d

    if "vertical_wall_1d" in _data:
        for vertical_wall_1d_data in _data.pop("vertical_wall_1d"):
            _id, vertical_wall_1d = VerticalWall1DCreator.create(vertical_wall_1d_data, constants)
            assert _id not in id_body_map, (list(id_body_map.keys()), _id)
            id_body_map[_id] = vertical_wall_1d

    # parse force information

    forces: list[ForceBase] = list()

    if "spring" in _data:
        forces.extend(
            [
                SpringCreator.create(spring_data, id_body_map, constants)
                for spring_data in _data.pop("spring")
            ]
        )

    if "gravity_like" in _data:
        forces.extend(
            [
                GravityLikeCreator.create(gravity_like_data)
                for gravity_like_data in _data.pop("gravity_like")
            ]
        )

    if "frictional_force_2d" in _data:
        forces.extend(
            [
                FrictionalForce2DCreator.create(frictional_force_2d, constants)
                for frictional_force_2d in _data.pop("frictional_force_2d")
            ]
        )

    if "non_sticky_left_horizontal_spring" in _data:
        forces.extend(
            [
                NonStickyLeftHorizontalSpringCreator.create(
                    non_sticky_left_horizontal_spring, constants
                )
                for non_sticky_left_horizontal_spring in _data.pop(
                    "non_sticky_left_horizontal_spring"
                )
            ]
        )

    if "horizontal_frictional_force_1d" in _data:
        forces.extend(
            [
                HorizontalFrictionalForce1DCreator.create(horizontal_frictional_force_1d, constants)
                for horizontal_frictional_force_1d in _data.pop("horizontal_frictional_force_1d")
            ]
        )

    if len(_data) > 0:
        logger.error(
            f"These field{'s are' if len(_data) > 1 else ' is'} not recognized:"
            + ", ".join(_data.keys())
        )
        sys.exit(1)

    return simulation_setting, Bodies(*id_body_map.values()), Forces(*forces)
