"""
utils
"""

from dynamics.force.forces import Forces
from dynamics.body.bodies import Bodies


def energy_info_text(bodies: Bodies, forces: Forces) -> list[str]:
    ke: float = bodies.kinetic_energy
    bpe: float = bodies.potential_energy(forces)
    fpe: float = forces.potential_energy
    pe: float = bpe + fpe
    de: float = bodies.dissipated_energy
    return [
        f"ke: {ke:.2f}, pe: {pe:.2f} (= bpe: {bpe:.2f} + fpe: {fpe:.2f}), de: {de:.2f}",
        f"ke + pe + de: {ke+pe+de:.2f}, ke + pe: {ke+pe:.2f}, pe: {pe:.2f}",
    ]
