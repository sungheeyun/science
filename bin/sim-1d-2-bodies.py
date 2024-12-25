"""
1-d simulate dynamics of two balls with spring connecting thw two
"""

import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: F401
from matplotlib.artist import Artist

from dynamics.bodies.point_mass import PointMass
from dynamics.bodies.bodies import Bodies
from dynamics.forces.forces import Forces
from dynamics.forces.horizontal_frictional_force_1d import HorizontalFrictionalForce1D
from dynamics.forces.spring import Spring
from dynamics.utils import energy_info

if __name__ == "__main__":

    # bodies
    ball_1: PointMass = PointMass(1, (-1, 0), (2, 0))
    ball_2: PointMass = PointMass(1.5, (1, 0), (-2, 0))

    bodies: Bodies = Bodies(ball_1, ball_2)

    # forces
    spring: Spring = Spring(10, 3, ball_1, ball_2)
    friction: HorizontalFrictionalForce1D = HorizontalFrictionalForce1D(0.1, 3)
    # gravity: GravityLike = GravityLike([-1.0, 0])

    # forces: Forces = Forces(spring, friction, gravity)
    forces: Forces = Forces(spring, friction)

    # forces.approx_min_energy(bodies)

    forces.register_forces(bodies)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    bodies.add_objs(ax)
    forces.add_objs(ax)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-1.0, 1.0)
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

    # Add time display
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    objs: list[Artist] = list(forces.objs) + list(bodies.objs) + [info_text]

    def init():
        """Initialize animation"""
        # ball.center = (0, 0)
        info_text.set_text("")

        return objs

    def animate(frame):
        """Animation function"""
        t = frame * 0.040  # Convert frame number to time (seconds)

        bodies.update(t, forces)
        forces.update_objs()

        info_text.set_text(
            f"@ {t:.2f} sec. - frame: {frame}" + "\n" + "\n".join(energy_info(bodies, forces)[0])
        )

        return objs

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=5000, interval=40, blit=True, repeat=False
    )

    # writer = PillowWriter(fps=20)
    # anim.save("ball_motion.gif", writer=writer)

    plt.show()
