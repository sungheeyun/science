"""
utils
"""

from typing import Any
from logging import Logger, getLogger
import sys

import numpy as np
from numpy.linalg import norm
from matplotlib.axes import Axes

from dynamics.accessories.accessory_base import AccessoryBase
from dynamics.accessories.accessories import Accessories
from dynamics.bodies.body_base import BodyBase
from dynamics.bodies.bodies import Bodies
from dynamics.bodies.fixed_body_base import FixedBodyBase
from dynamics.forces.force_base import ForceBase
from dynamics.forces.forces import Forces
from dynamics.instant_creators.constants import Constants
from dynamics.instant_creators.fan_creator import FanCreator
from dynamics.instant_creators.point_mass_creator import PointMassCreator
from dynamics.instant_creators.vertical_wall_1d_creator import VerticalWall1DCreator
from dynamics.instant_creators.vertical_pin_2d_creator import VerticalPin2DCreator
from dynamics.instant_creators.spring_creator import SpringCreator
from dynamics.instant_creators.gravity_like_creator import GravityLikeCreator
from dynamics.instant_creators.electric_like_creator import ElectricForceLikeCreator
from dynamics.instant_creators.frictional_force_2d_creator import FrictionalForce2DCreator
from dynamics.instant_creators.non_sticky_left_horizontal_spring_creator import (
    NonStickyLeftHorizontalSpringCreator,
)
from dynamics.instant_creators.horizontal_frictional_force_1d_creator import (
    HorizontalFrictionalForce1DCreator,
)

logger: Logger = getLogger()

_SQUARE_X_COORDINATES: np.ndarray = np.array([0, 1, 1, 0], float)
_SQUARE_Y_COORDINATES: np.ndarray = np.array([0, 0, 1, 1], float)


def is_mac_os() -> bool:
    return sys.platform == "darwin"


def energy_and_momentum_info(
    bodies: Bodies, forces: Forces
) -> tuple[list[str], np.ndarray, tuple[np.ndarray, ...]]:
    ke: float = bodies.total_kinetic_energy
    _bpe: float = bodies.total_potential_energy(forces)
    nspe, fpe = forces.potential_energy
    bpe: float = nspe + _bpe
    pe: float = bpe + fpe
    de: float = bodies.total_dissipated_energy
    total_momentum: np.ndarray = bodies.total_momentum

    force_potential_energy_bar_vertices: np.ndarray = np.vstack(
        (_SQUARE_X_COORDINATES, bpe + fpe * _SQUARE_Y_COORDINATES)
    )
    kinetic_energy_bar_vertices: np.ndarray = np.vstack(
        (_SQUARE_X_COORDINATES, (bpe + fpe) + ke * _SQUARE_Y_COORDINATES)
    )
    dissipated_energy_bar_vertices: np.ndarray = np.vstack(
        (_SQUARE_X_COORDINATES, (bpe + fpe + ke) + de * _SQUARE_Y_COORDINATES)
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
            "p = (" + ", ".join([f"{x:.2f}" for x in total_momentum]) + ")",
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
        + ")"
        + ", $v$ = ("
        + ", ".join([f"{x:.2f}" for x in body.vel])
        + ")"
        + f", $\|v\|$ = {norm(body.vel):.2f}"  # noqa: W605
        + ", $p$ = ("  # noqa: W605
        + ", ".join([f"{x:.2f}" for x in body.momentum])
        + ")"
        + r" & $E_\mathrm{d}$ = "
        + f"{body.dissipated_energy:.2f}"
        for body in bodies.bodies
        if not isinstance(body, FixedBodyBase)
    ]


def load_dynamic_system_simulation_setting(
    data: dict[str, Any]
) -> tuple[
    dict[str, str | float | int | bool | list[int | float] | None], Bodies, Forces, Accessories
]:
    # data hierarchy check
    assert "name" in data
    # assert "constants" in data
    assert "point_mass" in data

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

    simulation_setting: dict[str, str | float | int | bool | list[int | float] | None] = (
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
    simulation_setting["upper_left_window_corner_coordinate"] = simulation_setting.get(
        "upper_left_window_corner_coordinate", None
    )
    simulation_setting["repeat"] = simulation_setting.get("repeat", False)
    simulation_setting["show_kinematics"] = simulation_setting.get("show_kinematics", False)

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

    # parse accessories

    accessories: list[AccessoryBase] = list()

    if "fan" in _data:
        accessories.extend(
            [FanCreator.create(fan_data, constants) for fan_data in _data.pop("fan")]
        )

    # parse body information

    id_body_map: dict[str, BodyBase] = dict()

    for point_mass_data in _data.pop("point_mass"):
        _id, point_mass = PointMassCreator.create(point_mass_data, constants)
        assert _id not in id_body_map, (list(id_body_map.keys()), _id)
        id_body_map[_id] = point_mass

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

    if "electric_force_like" in _data:
        forces.extend(
            [
                ElectricForceLikeCreator.create(electric_like_data, id_body_map, constants)
                for electric_like_data in _data.pop("electric_force_like")
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

    return (
        simulation_setting,
        Bodies(*id_body_map.values()),
        Forces(*forces),
        Accessories(*accessories),
    )


# visualization


def remove_axes_boundary(ax: Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
