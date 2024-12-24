"""
simulation dynamics of rigid bodies, springs, gravity(-like), frictional forces, etc.
"""

from click import command, argument, Path
import yaml
from logging import Logger, getLogger

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from freq_used.logging_utils import set_logging_basic_config

from dynamics.utils import load_dynamic_system_simulation_setting
from dynamics.utils import energy_info_text, kinematics_info_text
from dynamics.body.bodies import Bodies

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
    fig, ax = plt.subplots(figsize=simulation_setting["fig_size"])

    forces.add_objs(ax)
    bodies.add_objs(ax)

    ax.set_xlim(*simulation_setting["xlim"])  # type:ignore
    ax.set_ylim(*simulation_setting["ylim"])  # type:ignore
    ax.grid(simulation_setting["grid"])  # type:ignore
    ax.set_aspect("equal")

    info_text = ax.text(0.02, 0.9875, "", transform=ax.transAxes, va="top")

    objs: list[Artist] = list(forces.objs) + list(bodies.objs) + [info_text]

    def init():
        """Initialize animation"""
        # ball.center = (0, 0)
        info_text.set_text("")

        return objs

    def animate(frame):
        """Animation function"""
        t = frame * real_world_time_interval  # convert frame number to time in sec

        bodies.update(t, forces)
        forces.update_objs()

        info_text.set_text(
            f"{t:.2f} sec. - frame: {frame}"
            + "\n"
            + "\n".join(energy_info_text(bodies, forces)[0])
            + "\n"
            + "\n".join(kinematics_info_text(bodies))
        )

        return objs

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

        ax.set_title(
            str(simulation_setting["name"])
            + f" ({real_world_time_interval * num_frames:.1f} sec."
            + f", up to {num_frames_per_sec:g} fps"
            + f" & {num_frames_per_sec * real_world_time_interval:g}x)"
            + f"\n- initial total energy: {energy_info_text(bodies, forces)[1]:.2f}",
            pad=10,
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

        ax.set_title(
            str(simulation_setting["name"])
            + f" ({real_world_time_interval * num_frames:.1f} sec."
            + f", up to {1./real_world_time_interval:g} fps"
            + f" & {real_world_time_interval * 1000.0 / frame_interval:g}x)"
            + f"\n- initial total energy: {energy_info_text(bodies, forces)[1]:.2f}",
            pad=10,
        )

        plt.show()
        logger.info("animation COMPLETED")


if __name__ == "__main__":
    main()
