"""
1-d simulate dynamics of a ball where gravity, spring force, and frictional force is applied to it
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa:F401
from matplotlib.artist import Artist

from dynamics.force.forces import Forces
from dynamics.body.point_mass import PointMass
from dynamics.body.bodies import Bodies
from dynamics.force.non_sticky_left_horizontal_spring import NonStickyLeftHorizontalSpring
from dynamics.force.gravity_like import GravityLike
from dynamics.force.horizontal_frictional_force_1d import HorizontalFrictionalForce1D
from dynamics.utils import energy_info

if __name__ == "__main__":

    # objects
    ball: PointMass = PointMass(2, (1, 0), (-2, 0))
    bodies: Bodies = Bodies(ball)

    # force sources
    spring: NonStickyLeftHorizontalSpring = NonStickyLeftHorizontalSpring(
        20,
        0,
    )
    gravity: GravityLike = GravityLike((-3, 0))
    friction: HorizontalFrictionalForce1D = HorizontalFrictionalForce1D(1, 1)

    forces: Forces = Forces(spring, gravity, friction)

    # forces.approx_min_energy(bodies)

    forces.register_forces(bodies)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(9, 4.5))

    bodies.add_objs(ax)
    forces.add_objs(ax)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.grid(axis="x")

    # Set title and labels
    ax.set_title(
        f"{os.path.splitext(os.path.split(__file__)[1])[0]}"
        + f" - initial total energy: {energy_info(bodies, forces)[1].sum():.2f}",
        pad=10,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("potential energy (J)")

    # Add a line to represent the path of motion
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # display
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    # draw potential energy
    xp_1d: np.ndarray = np.linspace(-5, 5, 100)
    ax.plot(xp_1d, forces.x_potential_energy(ball, xp_1d) * 0.25, "k")

    lim_info: dict[str, tuple[float, float]] = dict(
        x_lim=(np.inf, -np.inf), v_x_lim=(np.inf, -np.inf)
    )

    objs: list[Artist] = list(forces.objs) + list(bodies.objs) + [info_text]

    def init():
        """Initialize animation"""
        # ball.center = (0, 0)
        info_text.set_text("")

        return objs

    def animate(frame):
        """Animation function"""
        t = frame * 0.050  # Convert frame number to time (seconds)

        bodies.update(t, forces)
        forces.update_objs()

        # Update time display
        x_loc: float = float(ball.loc[0])
        v_x_vel: float = float(ball.vel[0])

        x_min, x_max = lim_info["x_lim"]
        lim_info["x_lim"] = (min(x_min, x_loc), max(x_max, x_loc))
        x_min, x_max = lim_info["x_lim"]

        v_x_min, v_x_max = lim_info["v_x_lim"]
        lim_info["v_x_lim"] = (min(v_x_min, v_x_vel), max(v_x_max, v_x_vel))
        v_x_min, v_x_max = lim_info["v_x_lim"]

        info_text.set_text(
            f"@ {t:.2f} sec. - {frame}\n"
            + ", ".join([f"x: {x_loc:.2f} m", f"v_x: {v_x_vel:.3f} m/s"])
            + "\n"
            + ", ".join(
                [f"x_lim: ({x_min:.2f},{x_max:.2f})", f"v_x_lim: ({v_x_min:.2f},{v_x_max:.2f})"]
            )
            + "\n"
            + "\n".join(energy_info(bodies, forces)[0])
        )

        return objs

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=1000, interval=50, blit=True, repeat=False
    )

    plt.show()
