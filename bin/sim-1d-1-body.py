"""
1-d simulate dynamics of a ball where gravity, spring force, and frictional force is applied to it
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa:F401
from matplotlib.artist import Artist

from dynamics.force.forces import Forces
from dynamics.body.rigid_ball import RigidBall
from dynamics.body.bodies import Bodies
from dynamics.force.non_sticky_left_horizontal_spring import NonStickyLeftHorizontalSpring
from dynamics.force.gravity_like import GravityLike
from dynamics.force.horizontal_frictional_force_1d import HorizontalFrictionalForce1D

if __name__ == "__main__":

    # objects
    ball: RigidBall = RigidBall(2.0, (1, 0), (0, 0))
    bodies: Bodies = Bodies(ball)

    # force sources
    spring: NonStickyLeftHorizontalSpring = NonStickyLeftHorizontalSpring(
        10.0,
        0.0,
    )
    gravity: GravityLike = GravityLike((-3.0, 0))
    friction: HorizontalFrictionalForce1D = HorizontalFrictionalForce1D(0.3, 0)

    forces: Forces = Forces(spring, gravity, friction)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    bodies.add_objs(ax)
    forces.add_objs(ax)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(axis="x")

    # Set title and labels
    ax.set_title(os.path.splitext(os.path.split(__file__)[1])[0], pad=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("potential energy (J)")

    # ax.add_patch(ball)

    # Add a line to represent the path of motion
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # display
    info_text = ax.text(0.02, 0.90, "", transform=ax.transAxes, va="top")

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
        t = frame * 0.010  # Convert frame number to time (seconds)

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

        ke: float = bodies.kinetic_energy
        bpe: float = bodies.potential_energy(forces)
        fpe: float = forces.potential_energy

        info_text.set_text(
            f"{frame}\n "
            + ", ".join([f"x: {x_loc:.2f} m", f"v_x: {v_x_vel:.3f} m/s"])
            + f" @ {t:.3f} sec"
            + "\n "
            + ", ".join(
                [f"x_lim: ({x_min:.2f},{x_max:.2f})", f"v_x_lim: ({v_x_min:.2f},{v_x_max:.2f})"]
            )
            + f"\n energy: {ke + bpe + fpe:.2f} (ke: {ke:.2f} + bpe {bpe:.2f} + fpe: {fpe:.2f})"
            + f"\n pe = bpe + fpe = {bpe + fpe:.2f}"
        )

        return objs

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=5000, interval=1, blit=True, repeat=False
    )

    # writer = PillowWriter(fps=20)
    # anim.save("ball_motion.gif", writer=writer)

    plt.show()
