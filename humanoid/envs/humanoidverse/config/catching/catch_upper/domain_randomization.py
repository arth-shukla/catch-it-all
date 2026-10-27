from dataclasses import dataclass
from typing import Dict, Tuple

from humanoid.utils.dataclasses import class_factory, default_field


@dataclass
class DomainRandomizationConfig:
    use_random: bool = True
    obj_random: bool = True
    cam_random: bool = True
    sim_random: bool = True

    randomize_cam_pos_bias: bool = "${domain_rand.cam_random}"
    cam_pos_bias: bool = 0.01

    randomize_cam_latency: bool = "${domain_rand.cam_random}"
    cam_latency_range: Tuple[int, int] = (1, 1)

    ####################################################################################
    # Sim Rand
    ####################################################################################

    randomize_gravity: bool = "${domain_rand.sim_random}"
    gravity_update_s: Tuple[float, float] = (4, 12)
    gravity_rand_scale: Tuple[float, float] = (-0.5, 0.5)

    ####################################################################################

    ####################################################################################
    # Object Rand
    ####################################################################################

    randomize_obj_latency: bool = False  # "${domain_rand.obj_random}"
    obj_latency_range: Tuple[int, int] = (1, 1)

    randomize_obj_scale: bool = "${domain_rand.obj_random}"
    obj_scale_ranges: Dict[str, Tuple[float, float]] = default_field(
        {
            "cylinder": (0.5, 1.0),
            "block": (0.5, 1.25),
            "default": (0.75, 1.5),
            "cuboid": (0.5, 1.0),
            "cuboid22": (0.5, 1.0),
            "apple": (0.75, 1.0),
            "flashlight": (1.25, 2.0),
            #
            # "book": (0.25, 1.15),
            # "hammer": (0.25, 1.15),
            # "dinner_plate": (0.25, 1.15),
            # "rubberduck": (0.75, 2.0),
            # "smallBall": (0.75, 2.0),
            # "apple": (0.75, 2.0),
            # "flashlight": (1.25, 3.0),
            "default": (0.5, 1.5),
        }
    )

    randomize_obj_mass: bool = "${domain_rand.obj_random}"
    obj_mass_scale_range: Tuple[float, float] = (0.5, 1.5)

    randomize_obj_friction: bool = "${domain_rand.obj_random}"
    obj_friction_range: Tuple[float, float] = (0.1, 1.2)

    randomize_obj_restitution: bool = "${domain_rand.obj_random}"
    obj_restitution_range: Tuple[float, float] = (0.0, 1.0)

    ####################################################################################

    ####################################################################################
    # Joint Pos Rand
    ####################################################################################

    # actuation offset
    randomize_actuation_offset: bool = "${domain_rand.use_random}"
    actuation_offset_range: Tuple[float, float] = (-0.05, 0.05)

    # randomize initial dof pos
    randomize_initial_joint_pos: float = "${domain_rand.use_random}"
    initial_joint_pos_scale: Tuple[float, float] = (0.8, 1.2)
    initial_joint_pos_offset: Tuple[float, float] = (-0.1, 0.1)

    ####################################################################################

    ####################################################################################
    # Ctrl Rand
    ####################################################################################

    # rfi
    randomize_torque_rfi: bool = "${domain_rand.use_random}"
    rfi_lim: float = 0.025
    randomize_rfi_lim: bool = False  # "${domain_rand.randomize_torque_rfi}"
    rfi_lim_range: Tuple[float, float] = (1.0, 1.0)

    # pd gain
    randomize_pd_gain: bool = "${domain_rand.use_random}"
    kp_range: Tuple[float, float] = (0.6, 1.25)
    kd_range: Tuple[float, float] = (0.6, 1.25)

    # control delay
    randomize_ctrl_delay: bool = "${domain_rand.use_random}"
    ctrl_delay_step_range: Tuple[int, int] = (0, 1)
    # integer max real delay is 90ms for (0, 2)

    ####################################################################################

    ####################################################################################
    # Robot Mass/Friction/Restitution
    ####################################################################################

    # torso/hand payload mass
    randomize_payload_mass: bool = False  # "${domain_rand.use_random}"
    payload_mass_range: Tuple[float, float] = (-5, 10)
    hand_payload_mass_range: Tuple[float, float] = (-0.1, 0.3)

    # link mass
    randomize_link_mass: bool = "${domain_rand.use_random}"
    link_mass_range: Tuple[float, float] = (0.8, 1.2)

    # friction restitution
    reset_friction_restitution_each_episode: bool = "${domain_rand.use_random}"
    randomize_friction: bool = "${domain_rand.use_random}"
    friction_range: Tuple[float, float] = (0.5, 1.25)
    randomize_restitution: bool = False  # "${domain_rand.use_random}"
    restitution_range: Tuple[float, float] = (0.0, 1.0)

    ####################################################################################

    ####################################################################################
    # Whole Body Rand
    ####################################################################################

    # randomize body displacement
    randomize_base_com: bool = False

    @dataclass
    class BaseComRange:
        x: Tuple[float, float] = (-0.1, 0.1)
        y: Tuple[float, float] = (-0.1, 0.1)
        z: Tuple[float, float] = (-0.1, 0.1)

    base_com_range: BaseComRange = class_factory(BaseComRange)

    # push robots
    push_robots: bool = False
    push_interval_s: Tuple[float, float] = (4, 5)
    max_push_vel_xy: float = 0.5

    ####################################################################################
