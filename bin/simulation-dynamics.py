"""
simulation dynamics of rigid bodies, springs, gravity(-like), frictional forces, etc.
"""

from click import command, argument, option, Path
import yaml
import os

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: F401
from matplotlib.artist import Artist

from dynamics.utils import load_dynamic_system_simulation_setting
from dynamics.utils import energy_info_text


@command()
@argument("input_file", type=Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@option(
    "-m",
    "--min-energy-init",
    is_flag=True,
    help="Set the initial body locations by (approximately) solving energy minimization problem",
)
def main(input_file: str, min_energy_init: bool) -> None:
    with open(input_file, "r") as fid:
        data = yaml.safe_load(fid)

    for key, value in data.items():
        if isinstance(value, list):
            print(key)
            for v in value:
                assert isinstance(v, dict), v.__class__
                print("\t", v)
        else:
            print(key, value, value.__class__)

    name, bodies, forces, min_energy = load_dynamic_system_simulation_setting(data)

    if min_energy:
        forces.approx_min_energy(bodies)

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

    objs: list[Artist] = list(forces.objs) + list(bodies.objs) + [info_text]

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
    anim = FuncAnimation(  # noqa: F841
        fig, animate, init_func=init, frames=2000, interval=1, blit=True, repeat=False
    )

    # writer = PillowWriter(fps=25)
    # anim.save("ball_motion.gif", writer=writer)

    plt.show()


if __name__ == "__main__":
    main()
