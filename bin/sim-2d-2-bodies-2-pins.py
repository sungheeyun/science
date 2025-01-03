"""
1-d simulate dynamics of two balls with spring connecting thw two
"""

import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: F401
from matplotlib.artist import Artist

from dynamics.bodies.point_mass import PointMass
from dynamics.bodies.bodies import Bodies
from dynamics.bodies.vertical_pin_2d import VerticalPin2D
from dynamics.forces.forces import Forces
from dynamics.forces.gravity_like import GravityLike
from dynamics.forces.frictional_force_2d import FrictionalForce2D
from dynamics.forces.spring import Spring
from dynamics.utils import energy_and_momentum_info

if __name__ == "__main__":

    # bodies
    ball_1: PointMass = PointMass(1, (-2, -1), (0, 0))
    ball_2: PointMass = PointMass(1, (2, 1), (0, 0))

    # pins
    pin_1: VerticalPin2D = VerticalPin2D((-2, -4))
    pin_2: VerticalPin2D = VerticalPin2D((2, 2))

    bodies: Bodies = Bodies(ball_1, ball_2, pin_2, pin_1)

    # forces
    spring_1: Spring = Spring(10, 10**0.2, pin_1, ball_1)
    spring_2: Spring = Spring(10, 10**0.2, ball_1, ball_2)
    spring_3: Spring = Spring(10, 10**0.2, ball_2, pin_2)

    friction: FrictionalForce2D = FrictionalForce2D(10**0.0, (2, 2))
    gravity: GravityLike = GravityLike([-5, 5])

    forces: Forces = Forces(spring_1, spring_2, spring_3, friction, gravity)

    # forces.approx_min_energy(bodies)

    forces.register_forces(bodies)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    bodies.add_objs(ax)
    forces.add_objs(ax)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.grid(True)

    # Set title and labels
    ax.set_title(
        f"{os.path.splitext(os.path.split(__file__)[1])[0]}"
        + f" - initial total energy: {energy_and_momentum_info(bodies, forces)[1].sum():.2f}",
        pad=10,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Add time display
    info_text = ax.text(0.02, 0.9875, "", transform=ax.transAxes, va="top")

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
            f"{t:.2f} sec. - frame: {frame}"
            + "\n"
            + f"xy_1: {ball_1.loc_text}, x_2: {ball_2.loc_text}"
            + f", v_x_1: {ball_1.vel_text}, v_x_2: {ball_2.vel_text}"
            + "\n"
            + "\n".join(energy_and_momentum_info(bodies, forces)[0])
        )

        return objs

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=250, interval=40, blit=True, repeat=False
    )

    # writer = PillowWriter(fps=20)
    # anim.save("ball_motion.gif", writer=writer)

    plt.show()
