from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from humanoid.utils.dataclasses import class_factory, default_field


@dataclass
class CatchEnvConfig:
    num_envs: int = 4096
    headless: bool = True
    record_video: bool = False
    render_mode: Optional[Literal["rgb_array", "human"]] = None

    env_spacing: float = 5.0
    max_episode_steps: int = 1000

    init_at_random_ep_len: bool = True

    toss_type: Literal["underhand", "overhand", "both"] = "underhand"

    toss_noise_curriculum: bool = False
    toss_noise_initial: float = 0.1
    toss_noise_min: float = 0.1
    toss_noise_max: float = float("inf")
    toss_noise_level_up_threshold: int = 85
    toss_noise_degree: float = 1e-5

    toss_lims: Dict[str, Dict[str, float]] = default_field(
        {
            "underhand_right": {"x": (0.0, 0.45), "y": (-0.4, 0), "z": (-0.2, 0.2)},
            "overhand_right": {"x": (-0.1, 0.2), "y": (-0.38, -0.125), "z": (0.3, 0.5)},
        }
    )

    @dataclass
    class Normalization:
        clip_observations: float = 100.0

    @dataclass
    class Termination:
        ignore_termination: bool = False
        # processed_actions_lock_arm_post_term: bool = False

        terminate_when_close_to_dof_pos_limit: bool = False
        terminate_when_close_to_dof_vel_limit: bool = False
        terminate_when_close_to_torque_limit: bool = False

        # loco
        terminate_by_contact: bool = False  # problematic in isaacgym when g1+terrain
        terminate_by_gravity: bool = True
        terminate_by_low_height: bool = True

        # catch
        terminate_by_obj_in_hand_goal: bool = False
        terminate_by_obj_in_hand_goal_static: bool = False
        terminate_by_obj_near_hand_goal_obj_static: bool = False
        terminate_by_hand_goal_obj_dist: bool = True
        terminate_by_obj_height: bool = True

        @dataclass
        class Scales:
            termination_close_to_dof_pos_limit: float = 0.98
            termination_close_to_dof_vel_limit: float = 0.98
            termination_close_to_torque_limit: float = 0.999

            # loco
            termination_min_base_height: float = 0.1
            termination_gravity_x: float = 0.8
            termination_gravity_y: float = 0.8

            # catch
            termination_hand_goal_obj_dist: float = 5.0
            termination_obj_height: float = (
                "${env.projectiles.obj_on_floor_threshold}"  # 0.2
            )

        scales: Scales = class_factory(Scales)

    @dataclass
    class Projectiles:
        obj_in_hand_goal_dist_threshold: float = 0.1
        obj_near_hand_goal_dist_threshold: float = (
            0.2  # hand_radius_threshold = 1.0 in dyanmic tasks
        )
        hand_static_threshold: float = 0.2
        obj_static_threshold: float = 1.25

        obj_on_floor_threshold: float = 0.2
        obj_height_max_threshold: float = 1.5

        ensure_obj_in_frame: bool = False

        @dataclass
        class ObjMetadata:
            asset_path: str
            center_of_mass: Tuple[float, float, float]
            dims: Tuple[float, float, float]

        included_objs: List[str] = default_field(
            [
                "apple",  # 0
                # "banana", # 2
                # "binoculars", # 3
                "block",  # 0
                # "book", # 2
                "cuboid",  # 0
                "cuboid22",  # 0
                "cylinder",  # 0
                # "dinner_plate", # 3
                "egg",  # 0
                "flashlight",  # 1
                "flute",  # 1
                # "hammer", # 3
                "lightbulb",  # 1
                # "mug", # 2
                "phone",  # 1
                # "pyramidmedium", # 3
                # "rubberduck", # 3
                "tennisBall",  # 0
                # "waterbottle", # 2
                "wristwatch",  # 1
            ]
        )
        obj_metadata_dict: Dict[str, ObjMetadata] = default_field(
            {
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/apple/apple.stl
                # com: (-0.00025178467753987047, 2.8882144232197185e-05, -0.006588340622658744)
                # dims (0.08677193, 0.08188984, 0.09850015)
                "apple": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/apple.xml",
                    center_of_mass=(0.0, 0.0, 0.0),
                    dims=(0.08677193, 0.08188984, 0.09850015),
                    # dims=(-0.04, 0.04, -0.04, 0.04, -0.04, 0.04),
                ),
                #
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/banana/banana.stl
                # com: (-0.004265362586410378, 0.016522619753314097, 0.000404023545257113)
                # dims (0.2049446, 0.096261695, 0.04514899)
                "banana": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/banana.xml",
                    center_of_mass=(-0.00426536, 0.01652262, 0.00040402),
                    dims=(0.2049446, 0.096261695, 0.04514899),
                    # dims=(-0.102474965, 0.10246963, -0.048129324, 0.048140377, -0.022573007, 0.022575982),
                ),
                #
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/binoculars/binoculars.stl
                # com: (1.262562895542367e-05, 0.00297264529311074, -0.003947296128095313)
                # dims (0.14164527, 0.047736205, 0.10899017)
                "binoculars": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/binoculars.xml",
                    center_of_mass=(0.0, 0.0, 0.0),
                    dims=(0.14164527, 0.047736205, 0.10899017),
                    # dims=(-0.2, 0.2, -0.2, 0.2, -0.005, 0.005),
                ),
                "block": ObjMetadata(
                    asset_path="urdf/objects/cube_multicolor_allegro.urdf",
                    center_of_mass=(0.0, 0.0, 0.0),
                    dims=(0.06, 0.06, 0.06),
                    # dims=(-0.06, 0.06, -0.06, 0.06, -0.06, 0.06),
                ),
                "book": ObjMetadata(
                    asset_path="cuboid/0000.urdf",
                    center_of_mass=(0.0, 0.0, 0.0),
                    dims=(0.23, 0.16, 0.03),
                    # dims=(-0.23, 0.23, -0.16, 0.16, -0.03, 0.03),
                ),
                "cuboid": ObjMetadata(
                    asset_path="throwObjects/cuboid/default/0035.urdf",
                    center_of_mass=(-0.00759802011, 5.84546466e-05, -0.000142452582),
                    dims=(0.076, 0.096, 0.08),
                    # dims=(-0.07425594, 0.074286126, -0.012241799, 0.012260404, -0.024747835, 0.024749983),
                ),
                "cuboid22": ObjMetadata(
                    asset_path="throwObjects/cuboid/default/0022.urdf",
                    center_of_mass=(0.0, 0.0, 0.0),
                    dims=(0.072, 0.08, 0.08),
                    # dims=(-0.04, 0.04, -0.04, 0.04, -0.04, 0.04),
                ),
                "cylinder": ObjMetadata(
                    asset_path="cylinder/default/0008.urdf",
                    center_of_mass=(0, 0, 0),
                    # urdf cylinder uses full-length + radius
                    dims=(0.04 * 2, 0.04 * 2, 0.096),
                    # dims=(-0.09600000000000003, 0.09600000000000003, -0.04, 0.04, -0.04, 0.04),
                ),
                # "dinner_plate": ObjMetadata(
                #     asset_path="dinner_plate/object.urdf",
                #     center_of_mass=(0.0, 0.0, 0.0),
                #     dims=(-0.2, 0.2, -0.2, 0.2, -0.005, 0.005),
                # ),
                "egg": ObjMetadata(
                    asset_path="mjcf/open_ai_assets/hand/egg.xml",
                    center_of_mass=(0.01, 0.01, 0.01),
                    # mjcf ellipsoid uses half-lengths
                    dims=(0.03 * 2, 0.03 * 2, 0.04 * 2),
                    # dims=(-0.03, 0.03, -0.03, 0.03, -0.04, 0.04),
                ),
                #
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/flashlight/flashlight.stl
                # com: (-2.540558735745884e-07, -0.00016339643846930186, -0.013782557087548325)
                # dims (0.033712544, 0.033758424, 0.13901761)
                "flashlight": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/flashlight.xml",
                    center_of_mass=(0.0, 0.0, 0.0),
                    dims=(0.033712544, 0.033758424, 0.13901761),
                    # dims=(-0.04, 0.04, -0.04, 0.04, -0.04, 0.04),
                ),
                #
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/flute/flute.stl
                # com: (-0.004224455876655573, -7.086969908761861e-07, -0.00017505204228872743)
                # dims (0.20998488, 0.016886517, 0.016907219)
                "flute": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/flute.xml",
                    center_of_mass=(0.0, 0.0, 0.0),
                    dims=(0.20998488, 0.016886517, 0.016907219),
                    # dims=(-0.2, 0.2, -0.2, 0.2, -0.005, 0.005),
                ),
                #
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/hammer/hammer.stl
                # com: (-0.00939875355525954, 0.014009893395745564, -4.697218775280076e-06)
                # dims (0.20498532, 0.11724059, 0.023202514)
                "hammer": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/hammer.xml",
                    center_of_mass=(-0.00939875356, 0.0140098934, -4.69721878e-06),
                    dims=(0.20498532, 0.11724059, 0.023202514),
                    # dims=(-0.10248412, 0.102501206, -0.058621638, 0.05861895, -0.011612458, 0.011590056),
                ),
                #
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/lightbulb/lightbulb.stl
                # com: (0.01450623327788068, 6.410755375491292e-06, 1.0786987198431091e-05)
                # dims (0.11045822, 0.06202586, 0.062029667)
                "lightbulb": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/lightbulb.xml",
                    center_of_mass=(0.0, 0.0, 0.0),
                    dims=(0.11045822, 0.06202586, 0.062029667),
                    # dims=(-0.04, 0.04, -0.04, 0.04, -0.04, 0.04),
                ),
                #
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/mug/mug.stl
                # com: (-0.013039771186918549, 1.8474289326681902e-05, -0.013635469626869344)
                # dims (0.11593024, 0.08225454, 0.10488628)
                "mug": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/mug.xml",
                    center_of_mass=(-0.0130397712, 1.84742893e-05, -0.0136354696),
                    dims=(0.11593024, 0.08225454, 0.10488628),
                    # dims=(-0.057947654, 0.057982583, -0.041128706, 0.04112583, -0.05247728, 0.052488506),
                ),
                #
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/phone/phone.stl
                # com: (-0.007598020107473099, 5.845464663174853e-05, -0.00014245258181533197)
                # dims (0.14853385, 0.024502203, 0.04949782)
                "phone": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/phone.xml",
                    center_of_mass=(0.0, 0.0, 0.0),
                    dims=(0.14853385, 0.024502203, 0.04949782),
                    # dims=(-0.07600000000000001, 0.07600000000000001, -0.09600000000000003, 0.09600000000000003, -0.08, 0.08),
                ),
                #
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/pyramidmedium/pyramidmedium.stl
                # com: (-4.35249078614016e-10, 1.6522464200545399e-09, -0.01999964212795397)
                # dims (0.07999857, 0.07999857, 0.07999857)
                "pyramidmedium": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/pyramidmedium.xml",
                    center_of_mass=(-4.35249079e-10, 1.65224642e-09, -0.0199996421),
                    dims=(0.07999857, 0.07999857, 0.07999857),
                    # dims=(-0.039999284, 0.03999928, -0.039999284, 0.039999288, -0.039999284, 0.039999284),
                ),
                #
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/rubberduck/rubberduck.stl
                # com: (0.00401011458435495, 1.5632234112054936e-06, -0.014328817215972246)
                # dims (0.09118228, 0.06678792, 0.0960367)
                "rubberduck": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/rubberduck.xml",
                    center_of_mass=(0.0, 0.0, 0.0),
                    dims=(0.09118228, 0.06678792, 0.0960367),
                    # dims=(-0.04, 0.04, -0.04, 0.04, -0.04, 0.04),
                ),
                "tennisBall": ObjMetadata(
                    asset_path="urdf/objects/tennis_ball.urdf",
                    center_of_mass=(0.0, 0.0, 0.0),
                    # urdf sphere uses radius
                    dims=(0.04 * 2, 0.04 * 2, 0.04 * 2),
                    # dims=(-0.04, 0.04, -0.04, 0.04, -0.04, 0.04),
                ),
                #
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/waterbottle/waterbottle.stl
                # com: (5.1175472762282614e-11, -1.4825729629895647e-10, -0.009460578326056752)
                # dims (0.06022059, 0.06022059, 0.1365)
                "waterbottle": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/waterbottle.xml",
                    center_of_mass=(5.11754728e-11, -1.48257296e-10, -0.00946057833),
                    dims=(0.06022059, 0.06022059, 0.1365),
                    # dims=(-0.030110296, 0.030110294, -0.030110296, 0.030110294, -0.06825, 0.06825),
                ),
                #
                # File: resources/projectiles/pregrasp_assets/assets/meshes/objects/wristwatch/wristwatch.stl
                # com: (0.010524666610117694, 9.76476772192088e-05, 0.00711449571318424)
                # dims (0.07525626, 0.046909273, 0.08241113)
                "wristwatch": ObjMetadata(
                    asset_path="pregrasp_assets/assets/objects/wristwatch.xml",
                    center_of_mass=(0.0, 0.0, 0.0),
                    dims=(0.07525626, 0.046909273, 0.08241113),
                    # dims=(-0.2, 0.2, -0.2, 0.2, -0.005, 0.005),
                ),
            }
        )

    normalization: Normalization = class_factory(Normalization)
    termination: Termination = class_factory(Termination)
    projectiles: Projectiles = class_factory(Projectiles)
