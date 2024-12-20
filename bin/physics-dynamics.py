"""
simulate dynamics in physics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from dynamics.force.forces import Forces
from dynamics.objs.rigid_ball import RigidBall
from dynamics.force.non_sticky_left_horizontal_spring import NonStickyLeftHorizontalSpring
from dynamics.force.const_force import ConstForce
from dynamics.force.horizontal_frictional_force import HorizontalFrictionalForce

if __name__ == "__main__":

    # objects
    rigid_ball: RigidBall = RigidBall(1.0, (1, 0), (-2, 0))

    # force sources
    spring: NonStickyLeftHorizontalSpring = NonStickyLeftHorizontalSpring(
        5.0,
        0.0,
    )
    gravity: ConstForce = ConstForce((-3.0, 0))
    friction: HorizontalFrictionalForce = HorizontalFrictionalForce(0.3, 0)

    forces: Forces = Forces(spring, gravity, friction)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    [ax.add_artist(obj) for obj in forces.objs]
    ax.add_patch(rigid_ball.obj)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(axis="x")

    # Set title and labels
    ax.set_title("one-dimensional rigid ball motion", pad=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("potential energy (J)")

    # ax.add_patch(ball)

    # Add a line to represent the path of motion
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Add time display
    info_text = ax.text(0.02, 0.85, "", transform=ax.transAxes)

    # draw potential energy
    xp_1d: np.ndarray = np.linspace(-5, 5, 100)
    ax.plot(xp_1d, forces.x_potential_energy(xp_1d) * 0.25, "k")

    def init():
        """Initialize animation"""
        # ball.center = (0, 0)
        info_text.set_text("")

        return forces.objs + [rigid_ball.obj, info_text]

    def animate(frame):
        """Animation function"""
        t = frame * 0.010  # Convert frame number to time (seconds)

        rigid_ball.update(t, forces)

        # Update time display
        info_text.set_text(
            f"{frame} - x: {rigid_ball.loc[0]:.2f} m, v_x: {rigid_ball.vel[0]:.3f} m/s @ {t:.3f} sec"
        )

        return forces.objs + [rigid_ball.obj, info_text]

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=5000, interval=2, blit=True, repeat=False
    )

    # writer = PillowWriter(fps=20)
    # anim.save("ball_motion.gif", writer=writer)

    plt.show()
