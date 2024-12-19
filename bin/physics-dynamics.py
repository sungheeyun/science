"""
simulate dynamics in physics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from dynamics.force.forces import Forces
from dynamics.objs.rigid_ball import RigidBall
from dynamics.force.left_horizontal_spring import LeftHorizontalSpring
from dynamics.force.const_force import ConstForce


# objects
rigid_ball: RigidBall = RigidBall(1.0, (2, 0))

# force sources
spring: LeftHorizontalSpring = LeftHorizontalSpring(
    1.0,
    0.0,
)
gravity: ConstForce = ConstForce((-1, 0))
forces: Forces = Forces(spring, gravity)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_xlim(-5, 5)
ax.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
ax.set_ylim(-1, 1)
ax.set_aspect("equal")
ax.grid(axis="x")

# Set title and labels
ax.set_title("one-dimensional rigid ball motion", pad=10)
ax.set_xlabel("Position (m)")
ax.set_ylabel("y")

ax.add_patch(rigid_ball.obj)
# ax.add_patch(ball)

num_coils = 25

t = np.linspace(0, num_coils * 2 * np.pi, 1000)

# Spring parameters
amplitude = 0.1  # Height of coils
x_stretch = 1.0  # Total length of spring = 1 unit

# Generate spring coordinates
x = np.linspace(-5, 0, len(t))  # x goes from -1 to 0
y = amplitude * np.sin(2 * np.pi * num_coils * (x / 5.0 + 1.0))  # Scaled to match x range

# Plot spring
# ax.plot(x, y, "b-", linewidth=2.0)
ax.plot(x, y, "b-", alpha=0.5)

# Add a line to represent the path of motion
ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

# Add time display
time_text = ax.text(0.02, 0.85, "", transform=ax.transAxes)

xp_1d: np.ndarray = np.linspace(-5, 5, 100)
ax.plot(xp_1d, forces.x_potential_energy(xp_1d) * 0.25, "k")


def init():
    """Initialize animation"""
    # ball.center = (0, 0)
    time_text.set_text("")

    # return ball, time_text
    return rigid_ball.obj, time_text


def animate(frame):
    """Animation function"""
    t = frame * 0.050  # Convert frame number to time (seconds)

    rigid_ball.update(t, forces)

    # Update time display
    time_text.set_text(f"time: {t:.1f} sec")

    # return time_text, ball
    return rigid_ball.obj, time_text


# Create animation
anim = FuncAnimation(
    fig, animate, init_func=init, frames=100000, interval=10, blit=True, repeat=False
)

plt.show()
