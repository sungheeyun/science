"""
1-d simulate dynamics of two balls with spring connecting thw two
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: F401
from matplotlib.artist import Artist

from dynamics.body.rigid_ball import RigidBall
from dynamics.body.bodies import Bodies
from dynamics.body.vertical_wall_1d import VerticalWall1D
from dynamics.force.forces import Forces
from dynamics.force.horizontal_frictional_force_1d import HorizontalFrictionalForce1D
from dynamics.force.spring import Spring
from dynamics.utils import energy_info_text

if __name__ == "__main__":

    Bodies.set_time_step_lengths(1e-3, 1e-3)

    # bodies
    wall_1: VerticalWall1D = VerticalWall1D(-3.0)
    wall_2: VerticalWall1D = VerticalWall1D(3.0)
    ball: RigidBall = RigidBall(1.0, (-1, 0), (0, 0))

    bodies: Bodies = Bodies(ball, wall_1, wall_2)

    # forces
    spring_1: Spring = Spring(2.0, 2.0, wall_1, ball)
    spring_2: Spring = Spring(5.0, 2.0, ball, wall_2)
    friction: HorizontalFrictionalForce1D = HorizontalFrictionalForce1D(0.0, 3)
    # gravity: GravityLike = GravityLike([-1.0, 0])

    forces: Forces = Forces(spring_1, spring_2, friction)

    forces.approx_min_energy(bodies)

    forces.register_forces(bodies)
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    bodies.add_objs(ax)
    forces.add_objs(ax)

    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(axis="x")

    # Set title and labels
    ax.set_title(
        f"{os.path.splitext(os.path.split(__file__)[1])[0]}"
        + f" - initial total energe: {energy_info_text(bodies,forces)[1]:.4f}",
        pad=10,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("potential energy (J)")

    # ax.add_patch(ball)

    # Add a line to represent the path of motion
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # draw potential energy
    xp_1d: np.ndarray = np.linspace(-5, 5, 100)
    ax.plot(xp_1d, (forces.x_potential_energy(ball, xp_1d) - 20.0) * 0.05, "k")

    # display
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

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

        info_text.set_text(
            f"{t:.2f} sec. - frame: {frame}"
            f", x: {ball.loc[0]:.2f}, v_x: {ball.vel[0]:.2f}"
            + "\n"
            + "\n".join(energy_info_text(bodies, forces)[0])
        )

        return objs

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=1000, interval=1, blit=True, repeat=False
    )

    # writer = PillowWriter(fps=20)
    # anim.save("sim-1d-2-bodies-2-walls.gif", writer=writer)

    plt.show()
