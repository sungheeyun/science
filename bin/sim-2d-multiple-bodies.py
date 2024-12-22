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
from dynamics.body.vertical_pin_2d import VerticalPin2D
from dynamics.force.forces import Forces
from dynamics.force.gravity_like import GravityLike
from dynamics.force.frictional_force_2d import FrictionalForce2D
from dynamics.force.spring import Spring
from dynamics.utils import energy_info_text

if __name__ == "__main__":

    Bodies.set_time_step_lengths(5e-3, 5e-3)

    # bodies
    ball_ul: RigidBall = RigidBall(1.0, (-2.0, 2.0))
    ball_ur: RigidBall = RigidBall(3.0, (2.0, 2.0))
    ball_ll: RigidBall = RigidBall(1.0, (-2.0, -2.0))
    ball_lr: RigidBall = RigidBall(4.0, (2.0, -2.0))

    ball_u: RigidBall = RigidBall(1.0, (0.0, 3.0))
    ball_r: RigidBall = RigidBall(1.0, (3.0, 0.0))

    # pins
    pin_ul: VerticalPin2D = VerticalPin2D((-4.0, 4.0))
    pin_ur: VerticalPin2D = VerticalPin2D((4.0, 4.0))
    pin_ll: VerticalPin2D = VerticalPin2D((-4.0, -4.0))
    pin_lr: VerticalPin2D = VerticalPin2D((4.0, -4.0))

    # bodies: Bodies = Bodies(ball_ul, ball_ur, ball_ll, ball_lr, pin_ul, pin_ur, pin_ll, pin_lr)
    bodies: Bodies = Bodies(
        ball_ul, ball_ur, ball_ll, ball_lr, pin_ul, pin_ur, pin_ll, pin_lr, ball_u, ball_r
    )

    common_spring_natural_length: float = 10.0 ** (-0.7)

    # forces
    spring_ul: Spring = Spring(10.0, common_spring_natural_length, pin_ul, ball_ul)
    spring_ur: Spring = Spring(5.0, common_spring_natural_length, pin_ur, ball_ur)
    spring_ll: Spring = Spring(10.0, common_spring_natural_length, pin_ll, ball_ll)
    spring_lr: Spring = Spring(7.0, common_spring_natural_length, pin_lr, ball_lr)

    spring_u: Spring = Spring(5.0, common_spring_natural_length, ball_ul, ball_ur)
    spring_l: Spring = Spring(7.0, common_spring_natural_length, ball_ll, ball_lr)
    spring_r: Spring = Spring(5.0, common_spring_natural_length, ball_lr, ball_ur)
    spring_le: Spring = Spring(5.0, common_spring_natural_length, ball_ll, ball_ul)

    # springs: list[Spring] = list()
    springs: list[Spring] = [
        Spring(10.0, common_spring_natural_length, pin_ul, ball_u, color="r", alpha=0.3),
        Spring(10.0, common_spring_natural_length, ball_u, ball_ur, color="r", alpha=0.3),
        Spring(10.0, common_spring_natural_length, ball_r, ball_ul, color="r", alpha=0.3),
        Spring(10.0, common_spring_natural_length, ball_r, ball_ll, color="r", alpha=0.3),
        Spring(10.0, common_spring_natural_length, ball_r, pin_lr, color="r", alpha=0.3),
        Spring(10.0, common_spring_natural_length, ball_r, pin_ur, color="r", alpha=0.3),
    ]

    friction: FrictionalForce2D = FrictionalForce2D(1e00, (2.0, 2.0))
    gravity: GravityLike = GravityLike([0.0, -5.0])

    forces: Forces = Forces(
        spring_ul,
        spring_ur,
        spring_ll,
        spring_lr,
        spring_u,
        spring_l,
        spring_r,
        spring_le,
        friction,
        gravity,
        *springs,
    )

    # forces.approx_min_energy(bodies)

    forces.register_forces(bodies)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    forces.add_objs(ax)
    bodies.add_objs(ax)

    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect("equal")
    ax.grid(True)

    # Set title and labels
    ax.set_title(
        f"{os.path.splitext(os.path.split(__file__)[1])[0]}"
        + f" - initial total energy: {energy_info_text(bodies,forces)[1]:.4f}",
        pad=10,
    )
    ax.set_xlabel("x (m)")
    ax.set_xlabel("y (m)")

    # Add time display
    info_text = ax.text(0.02, 0.9875, "", transform=ax.transAxes, va="top")

    lim_info: dict[str, tuple[float, float]] = dict(
        x_lim=(np.inf, -np.inf), v_x_lim=(np.inf, -np.inf)
    )

    objs: list[Artist] = list(forces.objs) + list(bodies.objs) + [info_text]
    updated_objs: list[Artist] = list(forces.updated_objs) + list(bodies.updated_objs) + [info_text]

    def init():
        """Initialize animation"""
        # ball.center = (0, 0)
        info_text.set_text("")

        return objs

    def animate(frame):
        """Animation function"""
        t = frame * 0.040  # Convert frame number to time in sec

        bodies.update(t, forces)
        forces.update_objs()

        info_text.set_text(
            f"{t:.2f} sec. - frame: {frame}" + "\n" + "\n".join(energy_info_text(bodies, forces)[0])
        )

        return objs

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=2000, interval=1, blit=True, repeat=False
    )

    # writer = PillowWriter(fps=25)
    # anim.save("ball_motion.gif", writer=writer)

    plt.show()

    # import cProfile
    #
    # profiler = cProfile.Profile()
    # profiler.enable()
    #
    # # cProfile.run("plt.show()")
    # plt.show()
    #
    # profiler.disable()
    #
    # profiler.dump_stats("aa.prof")
    #
    # import pstats
    #
    # stats = pstats.Stats("aa.prof")
    # # stats.sort_stats("cumulative").print_stats()
    # stats.sort_stats(1).print_stats()
