"""
1-d simulate dynamics of two balls with spring connecting thw two
"""

import os

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

    # bodies
    wall_1: VerticalWall1D = VerticalWall1D(-3)
    wall_2: VerticalWall1D = VerticalWall1D(3)
    ball_1: RigidBall = RigidBall(1, (-2, 0), (0, 0))
    ball_2: RigidBall = RigidBall(1, (2, 0), (0, 0))

    bodies: Bodies = Bodies(ball_1, ball_2, wall_1, wall_2)

    # forces
    spring_1: Spring = Spring(3.5, 1, wall_1, ball_1)
    spring_2: Spring = Spring(10, 1, ball_1, ball_2)
    spring_3: Spring = Spring(4.5, 1, ball_2, wall_2)
    friction: HorizontalFrictionalForce1D = HorizontalFrictionalForce1D(10.0 ** (0.0), 3)

    forces: Forces = Forces(spring_1, spring_2, spring_3, friction)

    # forces.approx_min_energy(bodies)

    forces.register_forces(bodies)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    bodies.add_objs(ax)
    forces.add_objs(ax)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.grid(axis="x")

    # Set title and labels
    ax.set_title(
        f"{os.path.splitext(os.path.split(__file__)[1])[0]}"
        + f" - initial total energy: {energy_info_text(bodies,forces)[1]:.2f}",
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
            f"@ {t:.2f} sec. - frame: {frame}"
            + f", x_1: {ball_1.loc[0]:.2f}, x_2: {ball_2.loc[0]:.2f}"
            + f", v_x_1: {ball_1.vel[0]:.2f}, v_x_2: {ball_2.vel[0]:.2f}"
            + "\n"
            + "\n".join(energy_info_text(bodies, forces)[0])
        )

        return objs

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=3000, interval=40, blit=True, repeat=False
    )

    # writer = PillowWriter(fps=20)
    # anim.save("sim-1d-2-bodies-2-walls.gif", writer=writer)

    plt.show()
