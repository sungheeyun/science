"""
simulation dynamics of rigid bodies, springs, gravity(-like), frictional forces, etc.
"""

from click import command, argument, Path
import yaml
from logging import Logger, getLogger

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Arrow
from matplotlib.text import Text
from freq_used.logging_utils import set_logging_basic_config
from freq_used.plotting import get_figure

from dynamics.body.bodies import Bodies
from dynamics.utils import load_dynamic_system_simulation_setting
from dynamics.utils import energy_info
from dynamics.utils import remove_axes_boundary

logger: Logger = getLogger()


@command()
@argument("input_file", type=Path(exists=True, file_okay=True, dir_okay=False, readable=True))
def main(input_file: str) -> None:
    set_logging_basic_config(__file__)

    with open(input_file, "r") as fid:
        data = yaml.safe_load(fid)

    simulation_setting, bodies, forces = load_dynamic_system_simulation_setting(data)

    Bodies.set_time_step_lengths(
        (
            simulation_setting["sim_time_step"]  # type:ignore
            if "sim_time_step" in simulation_setting
            else Bodies.SIM_TIME_STEP
        ),
        (
            simulation_setting["sim_time_step_const_vel"]  # type:ignore
            if "sim_time_step_const_vel" in simulation_setting
            else Bodies.SIM_TIME_STEP_CONST_VEL
        ),
    )

    if simulation_setting["minimize_energy"]:
        logger.info("set body locations as to (approximately) minimize the total potential energy")
        forces.approx_min_energy(bodies)

    forces.register_forces(bodies)

    frame_interval: float = float(simulation_setting["frame_interval"])  # type:ignore
    real_world_time_interval: float = float(
        simulation_setting["real_world_time_interval"]  # type:ignore
    )

    # Set up the figure and axis
    # fig, ax = plt.subplots(figsize=simulation_setting["fig_size"])
    xlim: list[float | int] = simulation_setting["xlim"]  # type:ignore
    ylim: list[float | int] = simulation_setting["ylim"]  # type:ignore
    x_range: float | int = xlim[1] - xlim[0]
    y_range: float | int = ylim[1] - ylim[0]
    window_width_inch: float | int = simulation_setting["window_width_inch"]  # type:ignore

    title_height_inch: float = 0.5
    info_text_head_height_cm: float = 1.5
    kinematics_info_height_cm: float = 0.0  # (2.8 / 5) * len(kinematics_info_text(bodies))
    info_text_box_height_inch: float = 0.393701 * (
        info_text_head_height_cm + kinematics_info_height_cm
    )
    energy_bar_width: float = 1.0
    energy_bar_padding: float | int = simulation_setting["energy_bar_padding"]  # type:ignore
    below_title_padding: float = (info_text_box_height_inch + 0.3) * 72

    fig: Figure = get_figure(
        1,
        2,
        axis_width=[window_width_inch, energy_bar_width],
        axis_height=window_width_inch * float(y_range) / float(x_range),
        left_margin=1.0,
        right_margin=1.5,
        bottom_margin=1.0,
        top_margin=below_title_padding / 72 + title_height_inch,
        horizontal_padding=energy_bar_padding,
    )
    animation_axis: Axes = fig.get_axes()[0]
    energy_bar_axis: Axes = fig.get_axes()[1]

    forces.add_objs(animation_axis)
    bodies.add_objs(animation_axis)

    animation_axis.set_xlim(*xlim)  # type:ignore
    animation_axis.set_ylim(*ylim)
    if isinstance(simulation_setting["grid"], str):
        animation_axis.grid(axis=simulation_setting["grid"])  # type:ignore
    else:
        animation_axis.grid(simulation_setting["grid"])  # type:ignore
    animation_axis.set_aspect("equal")

    if simulation_setting["1d"]:
        animation_axis.axhline(y=0.0, color="black", linestyle="-", alpha=0.5)

    info_text: Text = animation_axis.text(
        0.01,
        1.01,
        "",
        transform=animation_axis.transAxes,
        va="bottom",
    )

    remove_axes_boundary(animation_axis)

    (
        _,
        initial_energies,
        (
            force_potential_energy_bar_vertices,
            kinetic_energy_bar_vertices,
            dissipated_energy_bar_vertices,
        ),
    ) = energy_info(bodies, forces)
    initial_total_energy: float = initial_energies.sum()

    force_potential_energy_bar: Polygon = Polygon(
        force_potential_energy_bar_vertices.T, color="#cc9933", alpha=0.5
    )
    kinetic_energy_bar: Polygon = Polygon(kinetic_energy_bar_vertices.T, color="blue", alpha=0.5)
    dissipated_energy_bar: Polygon = Polygon(
        dissipated_energy_bar_vertices.T, color="red", alpha=0.5
    )

    arrow_width_ratio: float = 0.1

    body_potential_energy_arrow = Arrow(
        x=1.5,
        y=initial_energies[1],  # Start point
        dx=-0.4,
        dy=0,  # Direction and length
        color="black",
        alpha=0.5,
        width=arrow_width_ratio * np.diff(np.array(energy_bar_axis.get_ylim()))[0],
    )

    potential_energy_arrow = Arrow(
        x=1.5,
        y=initial_energies[1:3].sum(),  # Start point
        dx=-0.4,
        dy=0,  # Direction and length
        color="#cc9933",
        width=arrow_width_ratio * np.diff(np.array(energy_bar_axis.get_ylim()))[0],
    )

    energy_bar_axis.add_patch(force_potential_energy_bar)
    energy_bar_axis.add_patch(kinetic_energy_bar)
    energy_bar_axis.add_patch(dissipated_energy_bar)
    energy_bar_axis.add_patch(body_potential_energy_arrow)
    energy_bar_axis.add_patch(potential_energy_arrow)

    body_potential_energy_text: Text = energy_bar_axis.text(
        1.6, initial_energies[1], r"$E_\mathrm{p, gravity}$", ha="left", va="top"
    )
    potential_energy_text: Text = energy_bar_axis.text(
        1.6,
        initial_energies[1:3].sum(),
        r"$E_\mathrm{p, gravity}+E_\mathrm{p,spring}$",
        ha="left",
        va="top",
    )
    force_potential_energy_text: Text = energy_bar_axis.text(
        0.5,
        force_potential_energy_bar_vertices[1].mean(),
        r"$E_\mathrm{p,spring}$",
        ha="center",
        va="center",
    )
    kinetic_energy_text: Text = energy_bar_axis.text(
        0.5,
        kinetic_energy_bar_vertices[1].mean(),
        r"$E_\mathrm{k}$",
        ha="center",
        va="center",
    )
    dissipated_energy_text: Text = energy_bar_axis.text(
        0.5,
        dissipated_energy_bar_vertices[1].mean(),
        r"$E_\mathrm{d}$",
        ha="center",
        va="center",
    )

    energy_bar_y_lim: list[float] = [
        1.1 * initial_energies[1] - 0.1 * initial_total_energy,
        1.1 * initial_total_energy - 0.1 * initial_energies[1],
    ]

    energy_bar_axis.set_xlim(0, 1.5)
    energy_bar_axis.set_xticks([])
    energy_bar_axis.set_ylim(*energy_bar_y_lim)

    remove_axes_boundary(energy_bar_axis)

    def init():
        """Initialize animation"""
        # ball.center = (0, 0)
        info_text.set_text("")

        return []

    def animate(frame):
        """Animation function"""
        t = frame * real_world_time_interval  # convert frame number to time in sec

        bodies.update(t, forces)
        forces.update_objs()

        (
            energy_info_texts,
            energies,
            (
                force_potential_energy_bar_vertices,
                kinetic_energy_bar_vertices,
                dissipated_energy_bar_vertices,
            ),
        ) = energy_info(bodies, forces)

        if 1.1 * energies[1] - 0.1 * initial_total_energy < energy_bar_y_lim[0]:
            energy_bar_y_lim[0] = 1.1 * energies[1] - 0.1 * initial_total_energy
            energy_bar_axis.set_ylim(*energy_bar_y_lim)

        force_potential_energy_bar.set_xy(force_potential_energy_bar_vertices.T)
        kinetic_energy_bar.set_xy(kinetic_energy_bar_vertices.T)
        dissipated_energy_bar.set_xy(dissipated_energy_bar_vertices.T)

        body_potential_energy_arrow.set_data(
            y=energies[1],
            width=arrow_width_ratio * np.diff(np.array(energy_bar_axis.get_ylim()))[0],
        )
        body_potential_energy_text.set_y(energies[1])

        potential_energy_arrow.set_data(
            y=energies[1:3].sum(),
            width=arrow_width_ratio * np.diff(np.array(energy_bar_axis.get_ylim()))[0],
        )
        potential_energy_text.set_y(energies[1:3].sum())

        force_potential_energy_text.set_y(force_potential_energy_bar_vertices[1].mean())
        kinetic_energy_text.set_y(kinetic_energy_bar_vertices[1].mean())
        dissipated_energy_text.set_y(dissipated_energy_bar_vertices[1].mean())

        info_text.set_text(
            f"{t:.2f} sec. - frame: {frame}"
            + "\n"
            + "\n".join(energy_info_texts)
            # + "\n"
            # + "\n".join(kinematics_info_text(bodies))
        )

        return []

    num_frames: int = (
        simulation_setting["num_frames_saved"]  # type:ignore
        if simulation_setting["save_to_gif"]
        else simulation_setting["num_frames"]
    )

    # simulation

    logger.info(
        f"SIMULATION w/ sim_time_step: {Bodies.SIM_TIME_STEP}"
        + f" and sim_time_step_const_vel: {Bodies.SIM_TIME_STEP_CONST_VEL}"
    )
    logger.info(f"\t# total frames: {num_frames}")
    logger.info(
        f"\tone visualization update corresponds to {real_world_time_interval} sec. in real world"
    )
    logger.info(
        "\ttotal simulation corresponds to "
        + f"{real_world_time_interval * num_frames} sec. in real world"
    )

    anim = FuncAnimation(  # noqa: F841
        fig,
        animate,
        init_func=init,
        frames=num_frames,
        interval=frame_interval,
        blit=True,
        repeat=False,
    )

    if simulation_setting["save_to_gif"]:
        num_frames_per_sec: int = simulation_setting["num_frames_per_sec"]  # type:ignore
        gif_filepath: str = simulation_setting["gif_filepath"]  # type:ignore
        logger.info(
            f"Start saving {simulation_setting['num_frames_saved']} frames"
            + f" of dynamics simulation animation to {gif_filepath}"
            + f" with fps: {num_frames_per_sec}"
            + " ..."
        )
        logger.info(
            "\tthus .gif can be played (up to)"
            + f" {num_frames_per_sec * real_world_time_interval} times faster than real world"
        )

        animation_axis.set_title(
            str(simulation_setting["name"])
            + f" - {real_world_time_interval * num_frames:.1f} sec."
            + ", (up to) "
            + f"{num_frames_per_sec * real_world_time_interval:g}x"
            + " & "
            + f"{num_frames_per_sec:g} fps"
            + f"\ninitial total energy: {initial_total_energy:.2f}",
            pad=below_title_padding,
        )

        writer = PillowWriter(fps=num_frames_per_sec)
        anim.save(gif_filepath, writer=writer)
        logger.info("saving COMPLETED")
    else:
        logger.info("Start animation")
        logger.info(f"\t(try to) update visualization every {frame_interval / 1000.0:.3f} sec")
        logger.info(f"\tthat is, (try to) show {1000.0/frame_interval:.1f} frames per sec")
        logger.info(
            "\tthus the animation is (up to) "
            + f"{real_world_time_interval / frame_interval * 1000.0} times faster than real world"
        )

        animation_axis.set_title(
            str(simulation_setting["name"])
            + f" - {real_world_time_interval * num_frames:.1f} sec."
            + ", (up to) "
            + f"{real_world_time_interval * 1000.0 / frame_interval:g}x"
            + " & "
            + f"{1./real_world_time_interval:g} fps"
            + f"\ninitial total energy: {initial_total_energy:.2f}",
            pad=below_title_padding,
        )

        plt.show()
        logger.info("animation COMPLETED")


if __name__ == "__main__":
    main()
