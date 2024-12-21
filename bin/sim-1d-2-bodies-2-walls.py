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

if __name__ == "__main__":

    # bodies
    wall_1: VerticalWall1D = VerticalWall1D(-3.0)
    wall_2: VerticalWall1D = VerticalWall1D(3.0)
    ball_1: RigidBall = RigidBall(1.0, (-2, 0), (0, 0))
    ball_2: RigidBall = RigidBall(1.0, (2, 0), (0, 0))

    bodies: Bodies = Bodies(ball_1, ball_2, wall_1, wall_2)

    # forces
    spring_1: Spring = Spring(5.0, 1.5, wall_1, ball_1)
    spring_2: Spring = Spring(5.0, 2.5, ball_1, ball_2)
    spring_3: Spring = Spring(5.0, 1.5, ball_2, wall_2)
    friction: HorizontalFrictionalForce1D = HorizontalFrictionalForce1D(0.5, 3)
    # gravity: GravityLike = GravityLike([-1.0, 0])

    forces: Forces = Forces(spring_1, spring_2, spring_3, friction)

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

    # Add time display
    info_text = ax.text(0.02, 0.90, "", transform=ax.transAxes, va="top")

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

        info_text.set_text(
            f"{t:.2f} sec. - frame: {frame}"
            f", x_1: {ball_1.loc[0]:.2f}, x_2: {ball_2.loc[0]:.2f}"
            f", v_x_1: {ball_1.vel[0]:.2f}, v_x_2: {ball_2.vel[0]:.2f}"
        )

        return objs

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=200, interval=10, blit=True, repeat=False
    )

    # writer = PillowWriter(fps=20)
    # anim.save("sim-1d-2-bodies-2-walls.gif", writer=writer)

    plt.show()
