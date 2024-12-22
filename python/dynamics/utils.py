"""
utils
"""

from typing import Any

from dynamics.body.body_base import BodyBase
from dynamics.body.bodies import Bodies
from dynamics.force.force_base import ForceBase
from dynamics.force.forces import Forces
from dynamics.instant_creators.constants import Constants
from dynamics.instant_creators.rigid_ball_creator import RigidBallCreator
from dynamics.instant_creators.vertical_pin_2d_creator import VerticalPin2DCreator
from dynamics.instant_creators.spring_creator import SpringCreator
from dynamics.instant_creators.gtavity_like_creator import GravityLikeCreator
from dynamics.instant_creators.frictional_force_2d_creator import FrictionalForce2DCreator


def energy_info_text(bodies: Bodies, forces: Forces) -> tuple[list[str], float]:
    ke: float = bodies.kinetic_energy
    bpe: float = bodies.potential_energy(forces)
    fpe: float = forces.potential_energy
    pe: float = bpe + fpe
    de: float = bodies.dissipated_energy
    return [
        f"ke: {ke:.2f}, pe: {pe:.2f} (= bpe: {bpe:.2f} + fpe: {fpe:.2f}), de: {de:.2f}",
        f"ke + pe + de: {ke+pe+de:.4f}, ke + pe: {ke+pe:.2f}, pe: {pe:.2f}",
    ], ke + pe + de


def load_dynamic_system_simulation_setting(
    data: dict[str, Any]
) -> tuple[str, Bodies, Forces, bool]:
    # data hierarchy check
    assert "name" in data
    assert "constants" in data
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

    name: str = _data.pop("name")
    min_energy: bool = (
        bool(_data.pop("simulation_setting")["minimize_energy"])
        if "simulation_setting" in _data and "minimize_energy" in _data["simulation_setting"]
        else False
    )

    # parse constants
    constants: Constants = Constants()

    for key, value in _data.pop("constants").items():
        constants.assign(key, value)

    print("")
    print("CONSTANTS", "\n\t", constants)

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

    print("")
    print("BODYs")

    for _id, body in id_body_map.items():
        print("\t", _id, body)

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

    print("")
    print("FORCEs")

    for force in forces:
        print("\t", force)

    assert len(_data) == 0, (_data, list(_data.keys()))

    return name, Bodies(*id_body_map.values()), Forces(*forces), min_energy
