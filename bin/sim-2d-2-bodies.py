"""
1-d simulate dynamics of two balls with spring connecting thw two
"""

import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: F401
from matplotlib.artist import Artist

from dynamics.body.rigid_ball import RigidBall
from dynamics.body.bodies import Bodies
from dynamics.force.forces import Forces
from dynamics.force.gravity_like import GravityLike
from dynamics.force.frictional_force_2d import FrictionalForce2D
from dynamics.force.spring import Spring
from dynamics.utils import energy_info_text

if __name__ == "__main__":

    # bodies
    ball_1: RigidBall = RigidBall(1.0, (-2, -1), (0, 0))
    ball_2: RigidBall = RigidBall(1.0, (2, 1), (0, 0))

    bodies: Bodies = Bodies(ball_1, ball_2)

    # forces
    spring: Spring = Spring(10.0, 3.0, ball_1, ball_2)
    friction: FrictionalForce2D = FrictionalForce2D(1, (3.0, 1.5))
    gravity: GravityLike = GravityLike([0.0, 0])

    forces: Forces = Forces(spring, friction, gravity)

    # forces.approx_min_energy(bodies)

    forces.register_forces(bodies)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    bodies.add_objs(ax)
    forces.add_objs(ax)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-1.0, 2.0)
    ax.set_aspect("equal")
    ax.grid(True)

    # Set title and labels
    ax.set_title(
        f"{os.path.splitext(os.path.split(__file__)[1])[0]}"
        + f" - initial total energy: {energy_info_text(bodies,forces)[1]:.2f}",
        pad=10,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Add time display
    info_text = ax.text(0.02, 0.975, "", transform=ax.transAxes, va="top")

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
            + f", xy_1: {ball_1.loc_text}, x_2: {ball_2.loc_text}"
            + f", v_x_1: {ball_1.vel_text}, v_x_2: {ball_2.vel_text}"
            + "\n"
            + "\n".join(energy_info_text(bodies, forces)[0])
        )

        return objs

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=1000, interval=10, blit=True, repeat=False
    )

    # writer = PillowWriter(fps=20)
    # anim.save("ball_motion.gif", writer=writer)

    plt.show()
