from dataclasses import dataclass

from humanoid.utils.dataclasses import class_factory


@dataclass
class SimulatorConfig:
    name: str = "isaacgym"
    _target_: str = (
        "humanoid.envs.humanoidverse.simulator.isaacgym_catching.sim.IsaacGymCatchingSimulator"
    )

    @dataclass
    class PlaneConfig:
        static_friction: float = 1.0
        dynamic_friction: float = 1.0
        restitution: float = 0.0

    @dataclass
    class SimConfig:
        fps: int = 200
        control_decimation: int = 4
        substeps: int = 1

        @dataclass
        class PhysxConfig:
            num_threads: int = 4
            solver_type: int = 1  # 0: pgs, 1: tgs
            num_position_iterations: int = 8
            num_velocity_iterations: int = 8
            contact_offset: float = 0.002  # 0.01
            rest_offset: float = 0.0
            bounce_threshold_velocity: float = 0.01
            max_depenetration_velocity: float = 1000.0
            default_buffer_size_multiplier: float = 8.0
            max_gpu_contact_pairs: int = (
                "${robot.contact_pairs_multiplier} * 1024 * 1024"
            )
            contact_collection: int = 1

        physx: PhysxConfig = class_factory(PhysxConfig)

    plane: PlaneConfig = class_factory(PlaneConfig)
    sim: SimConfig = class_factory(SimConfig)

    def __post_init__(self):
        assert self.name == "isaacgym", "Wrong simulator cfg"
