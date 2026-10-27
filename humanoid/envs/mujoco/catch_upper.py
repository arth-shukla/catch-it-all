import copy
import datetime
import pickle
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from xml.dom import minidom

import cv2
import dacite
import mujoco
import mujoco.viewer
from line_profiler import profile

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch

from humanoid import HUMANOID_RESOURCES_DIR
from humanoid.envs.humanoidverse.config.catching.catch_upper import (
    G1CatchUpperLockLegsConfig,
    G1CatchUpperLockLegsHistConfig,
)
from humanoid.envs.humanoidverse.utils import torch_utils
from humanoid.envs.humanoidverse.utils.kinematics import Kinematics


PROJECTILE_BODY = "projectile"
PROJECTILE_JOINT = "projectile_freejoint"


def convert_urdf_to_mjcf_path(urdf_path: Path) -> Path:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))

    out = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
    out.close()

    mujoco.mj_saveLastXML(out.name, model)
    return Path(out.name)


def _strip_freejoints(elem: ET.Element):
    """Remove any object-side free joint so the wrapper freejoint is the only root DOF."""
    for parent in elem.iter():
        for child in list(parent):
            if child.tag == "freejoint":
                parent.remove(child)
            elif child.tag == "joint" and child.attrib.get("type") == "free":
                parent.remove(child)


def _clear_initial_body_pose(elem: ET.Element):
    """
    Object XMLs often store their standalone spawn pose on the top-level body.
    Once we wrap that body under PROJECTILE_BODY, that pose would become a
    permanent local offset. Clear it so PROJECTILE_BODY is the object root pose.
    """
    for attr in ("pos", "quat", "axisangle", "euler", "xyaxes", "zaxis"):
        elem.attrib.pop(attr, None)


def _copy_elem(elem: ET.Element) -> ET.Element:
    return copy.deepcopy(elem)


def _parse_mjcf_with_local_includes(xml_path: Path) -> ET.Element:
    """
    Parse an MJCF/XML file and expand its direct <include file=.../> tags
    relative to the file that contains the include.

    This is intentionally for object-side XMLs, not the robot XML. The robot is
    still included normally so its own asset/default structure is left intact.
    """
    xml_path = Path(xml_path).resolve()
    root = ET.parse(xml_path).getroot()

    def expand_children(parent: ET.Element, base_dir: Path):
        new_children = []
        for child in list(parent):
            if child.tag == "include":
                include_file = child.attrib.get("file")
                if include_file is None:
                    raise ValueError(f"<include> without file in {xml_path}")
                include_path = Path(include_file)
                if not include_path.is_absolute():
                    include_path = (base_dir / include_path).resolve()

                included_root = ET.parse(include_path).getroot()
                expand_children(included_root, include_path.parent)

                if included_root.tag == "mujoco":
                    new_children.extend([_copy_elem(x) for x in list(included_root)])
                else:
                    new_children.append(_copy_elem(included_root))
            else:
                expand_children(child, base_dir)
                new_children.append(child)
        parent[:] = new_children

    expand_children(root, xml_path.parent)
    return root


def _absolutize_asset_file_paths(asset_elem: ET.Element, asset_base_dir: Path):
    """
    The generated composed scene is written to /tmp, so relative mesh/texture
    paths from object XMLs would break. Make asset file paths absolute.
    """
    for elem in asset_elem.iter():
        file_attr = elem.attrib.get("file")
        if file_attr is None:
            continue
        file_path = Path(file_attr)
        if not file_path.is_absolute():
            elem.attrib["file"] = str((asset_base_dir / file_path).resolve())


def _collect_default_maps(root: ET.Element):
    """
    Collect MJCF defaults into:
      main_defaults[tag] -> attrs
      class_defaults[class_name][tag] -> attrs

    Handles the common MJCF pattern used by the projectile XMLs:
      <default>
        <geom .../>
        <default class="object_contact"><geom .../></default>
      </default>
    """
    main_defaults = {}
    class_defaults = {}

    def merge_defaults(default_elem: ET.Element, inherited, class_name):
        current = {tag: attrs.copy() for tag, attrs in inherited.items()}

        for child in list(default_elem):
            if child.tag == "default":
                continue
            current.setdefault(child.tag, {}).update(child.attrib)

        if class_name is None:
            for tag, attrs in current.items():
                main_defaults.setdefault(tag, {}).update(attrs)
        else:
            class_defaults[class_name] = {
                tag: attrs.copy() for tag, attrs in current.items()
            }

        for child in list(default_elem):
            if child.tag != "default":
                continue
            child_class = child.attrib.get("class", class_name)
            merge_defaults(child, current, child_class)

    for default_elem in root.findall("default"):
        merge_defaults(default_elem, {}, default_elem.attrib.get("class"))

    return main_defaults, class_defaults


def _apply_object_defaults(elem: ET.Element, main_defaults, class_defaults):
    """
    Bake object-side MJCF defaults into the object subtree, then remove known
    class attributes so the generated scene does not need to merge global
    object defaults that could accidentally affect the robot.
    """
    for node in elem.iter():
        tag_defaults = main_defaults.get(node.tag, {})
        for k, v in tag_defaults.items():
            node.attrib.setdefault(k, v)

        class_name = node.attrib.get("class")
        if class_name in class_defaults:
            class_tag_defaults = class_defaults[class_name].get(node.tag, {})
            for k, v in class_tag_defaults.items():
                node.attrib.setdefault(k, v)
            node.attrib.pop("class", None)


def _material_names(asset_children):
    names = set()
    for asset in asset_children:
        if asset.tag == "material" and "name" in asset.attrib:
            names.add(asset.attrib["name"])
    return names


def _referenced_material_names(body_children):
    names = set()
    for body in body_children:
        for node in body.iter("geom"):
            material = node.attrib.get("material")
            if material:
                names.add(material)
    return names


def _append_missing_materials(asset_children, body_children):
    """
    Some converted/legacy objects reference a material without defining it
    (e.g. material="material:object"). Add a neutral fallback material instead
    of failing during MuJoCo compile.
    """
    existing = _material_names(asset_children)
    referenced = _referenced_material_names(body_children)
    for name in sorted(referenced - existing):
        asset_children.append(
            ET.Element(
                "material",
                attrib={"name": name, "rgba": "0.8 0.8 0.8 1"},
            )
        )


def _parse_xyz_attr(elem: Optional[ET.Element], default: str = "0 0 0") -> str:
    if elem is None:
        return default
    return elem.attrib.get("xyz", default)


def _parse_rpy_attr(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is None:
        return None
    return elem.attrib.get("rpy")


def _normalize_urdf_mesh_scale(scale: Optional[str]) -> str:
    if scale is None:
        return "1 1 1"
    values = scale.split()
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError(f"URDF mesh scale must have 1 or 3 values, got {scale}")
    # Validate that these are numeric, but keep compact string formatting.
    return " ".join(str(float(x)) for x in values)


def _resolve_urdf_mesh_filename(filename: str, urdf_dir: Path) -> str:
    """Resolve URDF mesh filenames into absolute paths for generated MJCF.

    Supports absolute paths, relative paths, file:// URIs, and best-effort
    package:// URIs. For package://pkg/path, this searches ROS_PACKAGE_PATH
    and also walks upward from the URDF directory looking for a directory named
    pkg.
    """
    import os

    if filename.startswith("file://"):
        filename = filename[len("file://") :]

    if filename.startswith("package://"):
        package_rel = filename[len("package://") :]
        parts = Path(package_rel).parts
        if len(parts) < 2:
            raise ValueError(f"Invalid URDF package mesh filename: {filename}")
        package_name = parts[0]
        rel_inside_package = Path(*parts[1:])

        candidates = []
        ros_package_path = os.environ.get("ROS_PACKAGE_PATH", "")
        for base in [x for x in ros_package_path.split(os.pathsep) if x]:
            candidates.append(Path(base) / package_name / rel_inside_package)

        for parent in [urdf_dir] + list(urdf_dir.parents):
            candidates.append(parent / package_name / rel_inside_package)
            if parent.name == package_name:
                candidates.append(parent / rel_inside_package)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

        raise FileNotFoundError(
            f"Could not resolve URDF package mesh '{filename}'. "
            "Set ROS_PACKAGE_PATH or use relative/absolute mesh filenames."
        )

    path = Path(filename)
    if not path.is_absolute():
        path = urdf_dir / path
    return str(path.resolve())


def _urdf_geometry_to_mjcf_geom(
    geometry: ET.Element,
    *,
    visual: bool,
    urdf_dir: Path,
    asset_children,
    mesh_name: str,
) -> ET.Element:
    """
    Convert URDF geometry into an MJCF geom. For mesh geometry, add the
    corresponding <mesh> asset to asset_children and return a mesh geom.
    """
    geom_attrib = {}

    box = geometry.find("box")
    sphere = geometry.find("sphere")
    cylinder = geometry.find("cylinder")
    mesh = geometry.find("mesh")

    if box is not None:
        size = [float(x) for x in box.attrib["size"].split()]
        if len(size) != 3:
            raise ValueError(
                f"URDF box size must have 3 values, got {box.attrib['size']}"
            )
        geom_attrib.update(
            {
                "type": "box",
                "size": "{} {} {}".format(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0),
            }
        )
    elif sphere is not None:
        geom_attrib.update({"type": "sphere", "size": sphere.attrib["radius"]})
    elif cylinder is not None:
        # URDF cylinder length is full height; MJCF cylinder size is radius halfheight.
        radius = float(cylinder.attrib["radius"])
        length = float(cylinder.attrib["length"])
        geom_attrib.update({"type": "cylinder", "size": f"{radius} {length / 2.0}"})
    elif mesh is not None:
        filename = mesh.attrib.get("filename") or mesh.attrib.get("file")
        if not filename:
            raise ValueError("URDF mesh geometry missing filename/file attribute")

        mesh_file = _resolve_urdf_mesh_filename(filename, urdf_dir)
        mesh_scale = _normalize_urdf_mesh_scale(mesh.attrib.get("scale"))
        asset_children.append(
            ET.Element(
                "mesh",
                attrib={
                    "name": mesh_name,
                    "file": mesh_file,
                    "scale": mesh_scale,
                },
            )
        )
        geom_attrib.update({"type": "mesh", "mesh": mesh_name})
    else:
        raise ValueError(
            "Unsupported URDF geometry; expected box, sphere, cylinder, or mesh"
        )

    if visual:
        geom_attrib.update(
            {
                "name": mesh_name.replace("mesh", "visual"),
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
                "rgba": "0.8 0.2 0.2 1",
            }
        )
    else:
        geom_attrib.update(
            {
                "name": mesh_name.replace("mesh", "collision"),
                "condim": "4",
                "rgba": "0 1 0 0",
            }
        )

    return ET.Element("geom", attrib=geom_attrib)


def _extract_urdf_object_parts(urdf_path: Path):
    """
    Convert a simple URDF object directly into MJCF children that are appended
    *directly* to PROJECTILE_BODY. This keeps visual/collision geoms and mass on
    the same body that owns PROJECTILE_JOINT, which avoids zero-inertia wrapper
    and invisible point-only renders.
    """
    urdf_path = Path(urdf_path).resolve()
    root = ET.parse(urdf_path).getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected URDF <robot> root in {urdf_path}, got <{root.tag}>")

    links = root.findall("link")
    if not links:
        raise ValueError(f"No <link> found in URDF object {urdf_path}")
    if len(links) > 1:
        # Multi-link projectiles can be supported later by recursively converting
        # fixed joints, but your projectile assets are expected to be single rigid bodies.
        raise NotImplementedError(
            f"URDF object {urdf_path} has {len(links)} links. "
            "Direct runtime conversion currently supports single-link rigid objects."
        )

    link = links[0]
    asset_children = []
    projectile_children = []

    inertial = link.find("inertial")
    if inertial is not None:
        origin = inertial.find("origin")
        mass_elem = inertial.find("mass")
        inertia_elem = inertial.find("inertia")
        if mass_elem is not None and inertia_elem is not None:
            projectile_children.append(
                ET.Element(
                    "inertial",
                    attrib={
                        "pos": _parse_xyz_attr(origin),
                        "mass": mass_elem.attrib["value"],
                        "diaginertia": "{} {} {}".format(
                            inertia_elem.attrib.get("ixx", "1e-6"),
                            inertia_elem.attrib.get("iyy", "1e-6"),
                            inertia_elem.attrib.get("izz", "1e-6"),
                        ),
                    },
                )
            )

    for idx, visual in enumerate(link.findall("visual")):
        geometry = visual.find("geometry")
        if geometry is None:
            continue
        geom = _urdf_geometry_to_mjcf_geom(
            geometry,
            visual=True,
            urdf_dir=urdf_path.parent,
            asset_children=asset_children,
            mesh_name=f"projectile_visual_mesh_{idx}",
        )
        if idx and not geom.attrib.get("name", "").endswith(f"_{idx}"):
            geom.attrib["name"] = f"projectile_visual_{idx}"
        origin = visual.find("origin")
        geom.attrib["pos"] = _parse_xyz_attr(origin)
        rpy = _parse_rpy_attr(origin)
        if rpy is not None:
            geom.attrib["euler"] = rpy
        projectile_children.append(geom)

    for idx, collision in enumerate(link.findall("collision")):
        geometry = collision.find("geometry")
        if geometry is None:
            continue
        geom = _urdf_geometry_to_mjcf_geom(
            geometry,
            visual=False,
            urdf_dir=urdf_path.parent,
            asset_children=asset_children,
            mesh_name=f"projectile_collision_mesh_{idx}",
        )
        if idx and not geom.attrib.get("name", "").endswith(f"_{idx}"):
            geom.attrib["name"] = f"projectile_collision_{idx}"
        origin = collision.find("origin")
        geom.attrib["pos"] = _parse_xyz_attr(origin)
        rpy = _parse_rpy_attr(origin)
        if rpy is not None:
            geom.attrib["euler"] = rpy
        projectile_children.append(geom)

    if not any(child.tag == "geom" for child in projectile_children):
        raise ValueError(f"URDF object {urdf_path} produced no MJCF geoms")

    if not any(child.tag == "inertial" for child in projectile_children):
        # Fallback so the freejoint body is physical even if the URDF forgot inertial.
        projectile_children.insert(
            0,
            ET.Element(
                "inertial",
                attrib={
                    "pos": "0 0 0",
                    "mass": "0.05",
                    "diaginertia": "1e-4 1e-4 1e-4",
                },
            ),
        )

    return asset_children, projectile_children


def _extract_object_xml_parts(obj_path: Path):
    """
    Returns:
      asset_children: list[ET.Element]
      body_children: list[ET.Element]

    Handles:
      - full MJCF: <mujoco>...</mujoco>
      - body snippet: <body>...</body>
      - worldbody snippet: <worldbody>...</worldbody>

    Object-side <include> tags are expanded first, so files like apple.xml/mug.xml
    that include common.xml keep their assets/default classes.
    """
    obj_path = Path(obj_path).resolve()
    root = _parse_mjcf_with_local_includes(obj_path)

    asset_children = []
    body_children = []

    if root.tag == "mujoco":
        main_defaults, class_defaults = _collect_default_maps(root)

        for asset in root.findall("asset"):
            _absolutize_asset_file_paths(asset, obj_path.parent)
            asset_children.extend([_copy_elem(x) for x in list(asset)])

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"No <worldbody> found in {obj_path}")

        body_children.extend(
            [_copy_elem(x) for x in list(worldbody) if x.tag == "body"]
        )

    elif root.tag == "worldbody":
        main_defaults, class_defaults = {}, {}
        body_children.extend([_copy_elem(x) for x in list(root) if x.tag == "body"])

    elif root.tag == "body":
        main_defaults, class_defaults = {}, {}
        body_children.append(_copy_elem(root))

    else:
        raise ValueError(f"Unsupported object XML root <{root.tag}> in {obj_path}")

    for body in body_children:
        _clear_initial_body_pose(body)
        _strip_freejoints(body)
        _apply_object_defaults(body, main_defaults, class_defaults)

    _append_missing_materials(asset_children, body_children)
    return asset_children, body_children


def build_scene_with_object(
    robot_xml_path: Union[str, Path],
    obj_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Compose robot MJCF plus a runtime-selected projectile object.

    The object may be MJCF/XML or URDF. URDF is converted to MJCF first. The
    object is always wrapped as:

      <body name="projectile">
        <joint name="projectile_freejoint" type="free"/>
        ...object bodies/geoms...
        <site name="projectile_site"/>
      </body>

    This makes PROJECTILE_BODY/PROJECTILE_JOINT stable across all objects.
    """
    robot_xml_path = Path(robot_xml_path).resolve()
    obj_path = Path(obj_path).resolve()

    if obj_path.suffix.lower() == ".urdf":
        asset_children, body_children = _extract_urdf_object_parts(obj_path)
    elif obj_path.suffix.lower() == ".xml":
        asset_children, body_children = _extract_object_xml_parts(obj_path)
    else:
        raise ValueError(f"Expected .xml or .urdf object, got {obj_path}")

    mujoco_root = ET.Element("mujoco", attrib={"model": "robot_plus_projectile"})
    ET.SubElement(mujoco_root, "include", attrib={"file": str(robot_xml_path)})

    if asset_children:
        asset = ET.SubElement(mujoco_root, "asset")
        for child in asset_children:
            asset.append(child)

    worldbody = ET.SubElement(mujoco_root, "worldbody")
    projectile = ET.SubElement(
        worldbody,
        "body",
        attrib={
            "name": PROJECTILE_BODY,
            "pos": "0 0 1",
            "quat": "1 0 0 0",
        },
    )
    ET.SubElement(
        projectile,
        "joint",
        attrib={
            "name": PROJECTILE_JOINT,
            "type": "free",
        },
    )

    for body in body_children:
        projectile.append(body)

    ET.SubElement(
        projectile,
        "site",
        attrib={
            "name": "projectile_site",
            "pos": "0 0 0",
            "size": "0.01",
            "rgba": "1 0 0 1",
        },
    )

    if out_path is None:
        f = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
        f.close()
        out_path = Path(f.name)
    else:
        out_path = Path(out_path)

    rough_xml = ET.tostring(mujoco_root, encoding="unicode")
    pretty_xml = minidom.parseString(rough_xml).toprettyxml(indent="  ")
    out_path.write_text(pretty_xml)
    return out_path


@dataclass
class HumanoidConfig:
    sim_freq: int = 500
    control_freq: int = 50


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def to_tensor_64(x: np.ndarray):
    return torch.from_numpy(x).unsqueeze(0)


def to_tensor_32(x: np.ndarray):
    return torch.from_numpy(x).to(torch.float32).unsqueeze(0)


def to_numpy(x: torch.Tensor):
    return x.squeeze(0).to(torch.float64).numpy()


def tf_inverse(q, t):
    q_i, t_i = torch_utils.tf_inverse(to_tensor_64(q), to_tensor_64(t))
    return to_numpy(q_i), to_numpy(t_i)


def tf_apply(q, t, v):
    return to_numpy(
        torch_utils.tf_apply(to_tensor_64(q), to_tensor_64(t), to_tensor_64(v))
    )


def quat_apply(a, b):
    return to_numpy(torch_utils.quat_apply(to_tensor_64(a), to_tensor_64(b)))


def quat_rotate_inverse(q, v):
    return to_numpy(torch_utils.quat_rotate_inverse(to_tensor_64(q), to_tensor_64(v)))


def quat_mul(a, b):
    return to_numpy(torch_utils.quat_mul(to_tensor_64(a), to_tensor_64(b)))


def mj2ig_quat(q):
    return np.concatenate([q[1:], q[:1]])


def ig2mj_quat(q):
    return np.concatenate([q[3:], q[:3]])


class CatchUpperEnv(gym.Env):

    def __init__(self, render_mode="rgb_array", options_config={}, hist=False):
        super().__init__()

        self.mj_config = HumanoidConfig()

        self._tactile_obs = False
        self.config = (
            dacite.from_dict(G1CatchUpperLockLegsHistConfig, options_config)
            if hist
            else dacite.from_dict(G1CatchUpperLockLegsConfig, options_config)
        )

        self.used_obs_keys = set()
        all_obs_keys = []
        for key_list in self.config.obs.obs_dict.__dict__.values():
            all_obs_keys += key_list
        for key_dict in self.config.obs.obs_auxiliary.__dict__.values():
            all_obs_keys += list(key_dict.keys())
        for obs_key in all_obs_keys:
            raw_suffix = "_raw"
            if obs_key.endswith(raw_suffix):
                self.used_obs_keys.add(obs_key[: -len(raw_suffix)])
            self.used_obs_keys.add(obs_key)

        self.sim_freq = self.mj_config.sim_freq
        self.control_freq = self.mj_config.control_freq
        self.sim_timestep = round(1.0 / self.sim_freq, 5)
        self.control_timestep = round(1.0 / self.control_freq, 5)
        self.control_decimation = self.sim_freq // self.control_freq
        self.elapsed_steps = 0

        # Add metadata for supported render modes
        self.metadata = {
            "render_modes": ["human", "rgb_array", None],
            "render_fps": int(1.0 / self.sim_timestep),
        }
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize mujoco
        included_objs = self.config.env.projectiles.included_objs
        self.projectile_name = included_objs[self.np_random.choice(len(included_objs))]
        scene_xml = build_scene_with_object(
            robot_xml_path=self.config.robot.asset.asset_root
            / self.config.robot.asset.xml_file,
            obj_path=HUMANOID_RESOURCES_DIR
            / "projectiles"
            / self.config.env.projectiles.obj_metadata_dict[
                self.projectile_name
            ].asset_path,
        )

        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.sim_timestep

        self.default_dof_pos = np.array(
            [
                self.config.robot.init_state.default_joint_angles[dof_name]
                for dof_name in self.config.robot.dof_names
            ]
        )
        self.overhand_dof_pos = np.array(
            [
                self.config.robot.init_state.overhand_joint_angles[dof_name]
                for dof_name in self.config.robot.dof_names
            ]
        )
        self.dof_pos_limits_low = np.array(self.config.robot.dof_pos_lower_limit_list)
        self.dof_pos_limits_high = np.array(self.config.robot.dof_pos_upper_limit_list)
        self.torque_limits = np.array(self.config.robot.dof_effort_limit_list)

        self.toss_type = self.config.env.toss_type
        assert self.toss_type in ["underhand", "overhand"]
        self.hand = self.config.robot.used_hand
        self.arm_joint_indices = np.array(
            [
                self.config.robot.dof_names.index(name)
                for name in getattr(self.config.robot, f"{self.hand}_arm_joint_names")
            ]
        )
        self.finger_joint_indices = np.array(
            [
                self.config.robot.dof_names.index(name)
                for name in self.config.robot.finger_joint_names
            ]
        )
        self.finger_arm_joint_indices = np.concatenate(
            [self.finger_joint_indices, self.arm_joint_indices]
        )

        self.stiffness = np.zeros_like(self.dof_pos_limits_low)
        self.damping = np.zeros_like(self.dof_pos_limits_low)
        for i, dof_name in enumerate(self.config.robot.dof_names):
            for k, v in self.config.robot.control.stiffness.items():
                if k in dof_name:
                    self.stiffness[i] = v
                    break
            else:
                raise ValueError()
            for k, v in self.config.robot.control.damping.items():
                if k in dof_name:
                    self.damping[i] = v
                    break
            else:
                raise ValueError()

        assert np.all(
            np.abs(self.dof_pos_limits_low - self.model.jnt_range[:-1, 0]) < 1e-5
        )
        assert np.all(
            np.abs(self.dof_pos_limits_high - self.model.jnt_range[:-1:, 1]) < 1e-5
        )

        self._setup_gym_spaces()
        self._setup_obj_hand()

        self.history_buffer = None
        if hasattr(self.config.obs.obs_auxiliary, "history_actor"):
            self.history_buffer = dict(
                (k, np.zeros((v, self.config.obs.obs_dims[k])))
                for k, v in self.config.obs.obs_auxiliary.history_actor.items()
            )

        if self.config.robot.control.arm_control_mode == "ik":
            self._init_kinematics()

        self.renderer = None
        if self.render_mode == "human":
            self.renderer = mujoco.viewer.launch_passive(self.model, self.data)

        self.tracking_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "tracking_camera"
        )
        assert self.tracking_camera_id != -1, "tracking_camera not found in model"

        self.cam_config = self.config.camera
        assert (
            self.cam_config.width
            == self.model.cam_resolution[self.tracking_camera_id][0]
        )
        assert (
            self.cam_config.height
            == self.model.cam_resolution[self.tracking_camera_id][1]
        )
        assert (
            self.cam_config.vertical_fov == self.model.cam_fovy[self.tracking_camera_id]
        )

        self._camera_obs = any(
            x in self.used_obs_keys
            for x in [
                "obj_pose_wrt_camera",
                "obj_pos_wrt_camera_clean",
                "obj_pos_wrt_camera_noisy",
                "obj_pos_wrt_camera_noisy_sam2",
                "obj_lin_vel_wrt_camera",
                "obj_ang_vel_wrt_camera",
            ]
        )
        self._sam2_obs = "obj_pos_wrt_camera_noisy_sam2" in self.used_obs_keys
        assert not (
            self._sam2_obs and "obj_pos_wrt_camera_noisy" in self.used_obs_keys
        ), "use either obj_pos_wrt_camera_noisy or obj_pos_wrt_camera_noisy_sam2"
        self.tracking_renderer = None
        self.tracking_rgb = None
        self.obj_tracker = None
        if self._sam2_obs:
            from humanoid.object_tracking.sam2_tracker import SAM2ObjTracker

            self._init_renderer()
            self.obj_tracker = SAM2ObjTracker(
                fx=self.cam_config.fx,
                fy=self.cam_config.fy,
                cx=self.cam_config.cx,
                cy=self.cam_config.cy,
                model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
            )
        if self._camera_obs:
            self.cam_pos_queue = np.zeros(
                (self.config.domain_rand.cam_latency_range[1] + 1, 3)
            )
            self.cam_latency_idx = (
                self.config.domain_rand.cam_latency_range[0]
                if self.config.domain_rand.randomize_cam_latency
                else 0
            )
            self.cam_pos_bias = np.zeros(3)

        if self._tactile_obs:
            self._setup_tactile_sensors()
            self.tactile_forces = np.zeros(6, dtype=np.float32)  # 5 fingers + palm

    def _init_renderer(self):
        # Create renderer for tracking camera
        self.tracking_renderer = mujoco.Renderer(
            self.model,
            width=self.cam_config.width,
            height=self.cam_config.height,
        )
        self.tracking_renderer.disable_depth_rendering()
        self.tracking_rgb = None

    def _init_kinematics(self):
        self.kinematics = Kinematics(
            self.config.robot.asset.asset_root / self.config.robot.asset.urdf_file,
            "torso_link",
            f"{self.config.robot.used_hand}_wrist_yaw_link",
            dict(
                (name, i)
                for i, name in enumerate(
                    getattr(self.config.robot, f"{self.hand}_arm_joint_names")
                )
            ),
            getattr(self.config.robot, f"{self.hand}_arm_link_names"),
            "cpu",
        )
        eef_pos, eef_quat = self.kinematics.forward_kinematics(
            to_tensor_32(self.default_dof_pos[self.arm_joint_indices])
        )
        self.default_eef_pose = torch.cat([eef_pos, eef_quat], dim=1)
        self.target_eef_pose = self.default_eef_pose.clone()
        self.prev_target_eef_pose = self.target_eef_pose.clone()

        self.eef_pos_min = self.default_eef_pose[:, :3] + torch.tensor(
            list(self.ik_config.eef_pos_limits_lower),
        ).unsqueeze(0)
        self.eef_pos_max = self.default_eef_pose[:, :3] + torch.tensor(
            list(self.ik_config.eef_pos_limits_upper)
        ).unsqueeze(0)

        self._ik_dof_pos_noise_scale = self.config.obs.noise_scales[
            "dof_pos_finger_arm"
        ]
        self._default_ik_target_joint_angles = to_tensor_32(
            self.default_dof_pos[self.arm_joint_indices]
        )
        self._ik_target_joint_angles = self._default_ik_target_joint_angles.clone()
        self._prev_ik_target_joint_angles = self._default_ik_target_joint_angles.clone()
        self._prev_prev_ik_target_joint_angles = (
            self._default_ik_target_joint_angles.clone()
        )

    def ik_inverse_kinematics_v2(self, ik_pos_delta, ik_orn_delta, dof_pos_arm):
        target_pose = torch.cat([ik_pos_delta, ik_orn_delta])
        target_joint_angles, error = self.kinematics.inverse_kinematics(
            target_pose,
            dof_pos_arm,
            tol=self.ik_config.tol,
            max_iters=self.ik_config.max_iters,
            damping=self.ik_config.damping,
        )
        return target_joint_angles

    def _setup_gym_spaces(self):
        assert (
            self.config.robot.control.pos_controller_config.action_clip_value
            == "joint_limits"
        )
        self._act_limit_low = (
            (
                (self.dof_pos_limits_low - self.default_dof_pos)
                / self.config.robot.control.pos_controller_config.action_scale
            )
        ) * self.config.robot.control.pos_controller_config.action_lim_scale
        self._act_limit_high = (
            (
                (self.dof_pos_limits_high - self.default_dof_pos)
                / self.config.robot.control.pos_controller_config.action_scale
            )
        ) * self.config.robot.control.pos_controller_config.action_lim_scale

        finger_act_space_low = self._act_limit_low[self.finger_joint_indices].copy()
        finger_act_space_high = self._act_limit_high[self.finger_joint_indices].copy()

        if self.config.robot.control.arm_control_mode == "pos":
            arm_act_space_low = self._act_limit_low[self.arm_joint_indices].copy()
            arm_act_space_high = self._act_limit_high[self.arm_joint_indices].copy()
            self.act_space_low = np.concatenate(
                [finger_act_space_low, arm_act_space_low]
            )
            self.act_space_high = np.concatenate(
                [finger_act_space_high, arm_act_space_high]
            )
            act_space_dim = self.finger_joint_indices.size + self.arm_joint_indices.size
        elif self.config.robot.control.arm_control_mode == "ik":
            self.ik_config = self.config.robot.control.ik_controller_config
            arm_act_space_high = np.array(
                [self.ik_config.max_delta_pos * self.ik_config.delta_pos_action_scale]
                * 3
                + [self.ik_config.max_delta_rpy * self.ik_config.delta_rpy_action_scale]
                * 3,
            )
            arm_act_space_low = -arm_act_space_high.copy()

            self.act_space_low = np.concatenate(
                [finger_act_space_low, arm_act_space_low]
            )
            self.act_space_high = np.concatenate(
                [finger_act_space_high, arm_act_space_high]
            )
            act_space_dim = self.finger_joint_indices.size + 6
        else:
            assert self.config.robot.control.arm_control_mode == "target_delta_pos"
            assert (
                self.config.robot.control.target_delta_pos_controller_config.normalize_action_space
            )
            action_lim_scale = (
                self.config.robot.control.target_delta_pos_controller_config.action_lim_scale
            )
            arm_act_space_low = (
                -np.ones_like(self._act_limit_low[self.arm_joint_indices])
                * action_lim_scale
            )
            arm_act_space_high = (
                np.ones_like(self._act_limit_low[self.arm_joint_indices])
                * action_lim_scale
            )
            self.act_space_low = np.concatenate(
                [finger_act_space_low, arm_act_space_low]
            )
            self.act_space_high = np.concatenate(
                [finger_act_space_high, arm_act_space_high]
            )
            act_space_dim = self.finger_joint_indices.size + self.arm_joint_indices.size

        assert act_space_dim == self.config.robot.actions_dim

        self.action_space = spaces.Box(
            low=self.act_space_low.astype(np.float32),
            high=self.act_space_high.astype(np.float32),
            shape=(act_space_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.robot.policy_obs_dim,),
            dtype=np.float32,
        )

        self.action = np.zeros_like(self.default_dof_pos)

    def _setup_obj_hand(self):
        self.head_name = "head_link"
        self.head_site_idx = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.head_name
        )
        assert self.head_site_idx != -1

        self.hand_name = getattr(self.config.robot, f"{self.hand}_hand_name") + "_site"
        self.hand_site_idx = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.hand_name
        )
        assert self.hand_site_idx != -1
        self.hand_body_idx = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            getattr(self.config.robot, f"{self.hand}_hand_name"),
        )
        assert self.hand_body_idx != -1
        # same goal point as the isaacgym env, in the palm body frame
        self.hand_goal_point_in_hand = np.array([0.05, 0.0, -0.05])

        self.obj_body_idx = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, PROJECTILE_BODY
        )
        assert self.obj_body_idx != -1

        self.obj_joint_idx = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, PROJECTILE_JOINT
        )
        assert self.obj_joint_idx != -1

        self.obj_qpos_start = self.model.jnt_qposadr[self.obj_joint_idx]
        self.obj_qvel_start = self.model.jnt_dofadr[self.obj_joint_idx]

        self.projectile_scale = np.array([1.0])
        self.projectile_dims = np.array(
            self.config.env.projectiles.obj_metadata_dict[self.projectile_name].dims
        )
        self.projectile_dims_scaled = self.projectile_dims * self.projectile_scale

        self.torso_body_idx = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.config.robot.torso_name
        )

        self.toss_dist_ranges = [
            [1.0, 1.45],  # short
            [1.45, 2.0],  # med
            [2.0, 2.75],  # long
        ]

    def _setup_tactile_sensors(self):
        assert self._tactile_obs
        """Setup tactile sensor body indices for force sensing"""

        self.fsr_body_names = [
            "f1_finger_tip_fsr",
            "f2_finger_tip_fsr",
            "f3_finger_tip_fsr",
            "f4_finger_tip_fsr",
            "f5_finger_tip_fsr",
            f"{self.hand}_palm_fsr",
        ]

        # Get body indices
        self.fsr_body_indices = []
        for name in self.fsr_body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id == -1:
                print(f"Warning: FSR sensor body '{name}' not found, using -1")
            self.fsr_body_indices.append(body_id)

        self.fsr_body_indices = np.array(self.fsr_body_indices)

        # Parent body names
        self.fsr_parent_body_names = [
            "f1_finger_tip",
            "f2_finger_tip",
            "f3_finger_tip",
            "f4_finger_tip",
            "f5_finger_tip",
            f"{self.hand}_palm",
        ]

        self.fsr_parent_indices = []
        for name in self.fsr_parent_body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id == -1:
                print(f"Warning: FSR parent body '{name}' not found, using -1")
            self.fsr_parent_indices.append(body_id)

        self.fsr_parent_indices = np.array(self.fsr_parent_indices)

    def randomize_projectile_start(self):
        if self.config.env.projectiles.ensure_obj_in_frame:
            min_start_x, min_start_y, min_start_z = 0.5, -1.0, 0.25
            max_start_x, max_start_y, max_start_z = 2.75, 1.0, 2.0
        else:
            min_start_x, min_start_y, min_start_z = 0.5, -2.0, 0.25
            max_start_x, max_start_y, max_start_z = 2.75, 2.0, 2.0

        prob_short_toss = 0.4
        prob_med_toss = 0.4
        prob_long_toss = 0.2

        toss_rand_select = self._sample_random(0.0, 1.0)
        toss_type = 0
        if toss_rand_select > (prob_short_toss + prob_med_toss):
            toss_type = 2
        elif toss_rand_select > prob_short_toss:
            toss_type = 1

        toss_dist_range = self.toss_dist_ranges[toss_type]  # (num_envs, 1, 2)
        x_o = self._sample_random(toss_dist_range[0], toss_dist_range[1])
        y_o = self._sample_random(min_start_y, max_start_y)
        z_o = self._sample_random(min_start_z, max_start_z)

        return x_o, y_o, z_o

    def random_toss(self):
        # get obj init pos (relative to env origins)
        x_o, y_o, z_o = self.randomize_projectile_start()

        # get obj end pos (relative to env origins)
        robot_init_pos = self.data.xpos[self.torso_body_idx].copy()
        x_f, y_f, z_f = 0.0, 0.0, 0.8
        toss_lims = self.config.env.toss_lims[f"{self.toss_type}_{self.hand}"]
        x_f += self._sample_random(*toss_lims["x"])
        y_f += self._sample_random(*toss_lims["y"])
        z_f += self._sample_random(*toss_lims["z"])

        delta_x = x_f - x_o
        delta_y = y_f - y_o
        delta_z = z_f - z_o

        if delta_x < 1.0:
            t_impact = self._sample_random(0.4, 0.8)
        elif 1.0 <= delta_x <= 2.0:
            t_impact = self._sample_random(2.5, 0.8)
        else:  # elif 2.0 < delta_x:
            t_impact = self._sample_random(5.0, 3.5)

        # Solve velocity
        v_xo = delta_x / t_impact
        v_yo = delta_y / t_impact
        v_zo = (delta_z - 0.5 * (-9.81) * np.square(t_impact)) / t_impact

        # ang vel based on obj scale (larger scale -> less ang vel)
        obj_scale = self.projectile_scale
        if obj_scale < 0.85:
            obj_ang_vel = self._sample_random(-25, 25, size=3)
        elif 0.85 <= obj_scale <= 1.25:
            obj_ang_vel = self._sample_random(-15, 15, size=3)
        else:  # elif 1.25 < obj_scale:
            obj_ang_vel = self._sample_random(-12, 12, size=3)
        w_ox, w_oy, w_oz = obj_ang_vel

        return {
            "obj_init_pos": np.array([x_o, y_o, z_o]),
            "obj_init_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "obj_init_lin_vel": np.array([v_xo, v_yo, v_zo]),
            "obj_init_ang_vel": np.array([w_ox, w_oy, w_oz]),
            "obj_goal_pos": np.array([x_f, y_f, z_f]),
        }

    def _compute_tactile_forces(self):
        assert self._tactile_obs
        self.tactile_forces[:] = 0.0

        for i, parent_idx in enumerate(self.fsr_parent_indices):
            if parent_idx == -1:
                continue

            total_force = 0.0

            for j in range(self.data.ncon):
                contact = self.data.contact[j]

                if contact.geom1 == parent_idx or contact.geom2 == parent_idx:
                    c_array = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(self.model, self.data, j, c_array)

                    normal_force = abs(c_array[0])
                    total_force += normal_force

            self.tactile_forces[i] = 1.0 if total_force > 0.1 else 0.0

    def _sample_random(self, low, high, size=None):
        return self.np_random.random(size=size) * (high - low) + low

    def set_obj_init_state(self):
        self.data.qpos[self.obj_qpos_start : self.obj_qpos_start + 3] = self.obj_toss[
            "obj_init_pos"
        ]
        self.data.qpos[self.obj_qpos_start + 3 : self.obj_qpos_start + 7] = ig2mj_quat(
            self.obj_toss["obj_init_quat"]
        )
        self.data.qvel[self.obj_qvel_start : self.obj_qvel_start + 6] = np.concatenate(
            [self.obj_toss["obj_init_lin_vel"], self.obj_toss["obj_init_ang_vel"]]
        )
        self.obj_target_pos = self.obj_toss["obj_goal_pos"]
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if self.history_buffer is not None:
            for k, v in self.history_buffer.items():
                self.history_buffer[k] = v * 0

        if self._camera_obs:
            self.cam_pos_queue[:] = 0.0
            self.cam_latency_idx = (
                self.np_random.integers(
                    self.config.domain_rand.cam_latency_range[0],
                    self.config.domain_rand.cam_latency_range[1] + 1,
                )
                if self.config.domain_rand.randomize_cam_latency
                else 0
            )
            if not self._sam2_obs and self.config.domain_rand.randomize_cam_pos_bias:
                self.cam_pos_bias = self._sample_random(
                    -self.config.domain_rand.cam_pos_bias,
                    self.config.domain_rand.cam_pos_bias,
                    3,
                )
        if self._sam2_obs:
            self.obj_tracker.reset()

        # Clear GPU cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[: self.obj_qpos_start] = (
            self.default_dof_pos
            if self.toss_type == "underhand"
            else self.overhand_dof_pos
        )
        self.obj_toss = self.random_toss()
        self.set_obj_init_state()
        throw_wait_s = self._sample_random(0.1, 0.5)
        self.throw_wait_steps = np.array(
            throw_wait_s / self.control_timestep, dtype=np.int32
        )

        if self.config.robot.control.arm_control_mode == "ik":
            curr_dof_pos = to_tensor_32(self.data.qpos[: self.obj_qpos_start])[
                :, self.arm_joint_indices
            ]
            eef_pos, eef_quat = self.kinematics.forward_kinematics(curr_dof_pos)
            curr_eef_pose = torch.cat([eef_pos, eef_quat], dim=1)
            self.target_eef_pose[:] = curr_eef_pose
            self.prev_target_eef_pose[:] = curr_eef_pose
            self._ik_target_joint_angles[:] = curr_dof_pos
            self._prev_ik_target_joint_angles[:] = curr_dof_pos
            self._prev_prev_ik_target_joint_angles[:] = curr_dof_pos

        self.elapsed_steps = 0

        info = self.evaluate()
        self.target_dof_pos = info["robot_qpos"][self.arm_joint_indices].copy()
        self.action = np.zeros_like(info["robot_qpos"])
        return self._get_obs(info), info

    @profile
    def _rescale_arm_actions(self, arm_actions):
        if self.config.robot.control.arm_control_mode == "pos":
            return arm_actions

        if self.config.robot.control.arm_control_mode == "ik":
            arm_actions = to_tensor_32(arm_actions)

            self.prev_target_eef_pose[:] = self.target_eef_pose[:]
            self._prev_prev_ik_target_joint_angles[:] = (
                self._prev_ik_target_joint_angles[:]
            )
            self._prev_ik_target_joint_angles[:] = self._ik_target_joint_angles[:]

            pos_delta = arm_actions[:, :3] / self.ik_config.delta_pos_action_scale
            orn_rpy_delta = arm_actions[:, 3:] / self.ik_config.delta_rpy_action_scale

            orn_quat_delta = torch_utils.quat_from_euler_xyz(
                orn_rpy_delta[:, 0], orn_rpy_delta[:, 1], orn_rpy_delta[:, 2]
            )

            self.target_eef_pose[:, :3] = torch.clip(
                self.target_eef_pose[:, :3] + pos_delta,
                min=self.eef_pos_min,
                max=self.eef_pos_max,
            )
            self.target_eef_pose[:, 3:] = torch_utils.quat_mul(
                orn_quat_delta, self.target_eef_pose[:, 3:]
            )
            if self.ik_config.target_eef_pose_ema is not None:
                ema = self.ik_config.target_eef_pose_ema
                self.target_eef_pose[:, :3] = (
                    ema * self.prev_target_eef_pose[:, :3]
                    + (1.0 - ema) * self.target_eef_pose[:, :3]
                )
                self.target_eef_pose[:, 3:] = torch_utils.quat_weighted_average(
                    self.prev_target_eef_pose[:, 3:],
                    self.target_eef_pose[:, 3:],
                    ema,
                    1.0 - ema,
                )

            self._ik_dof_pos_arm_noisy = to_tensor_32(
                self.data.qpos[:-7][self.arm_joint_indices]
            ).clone()
            if self.ik_config.noised_dof_pos:
                self._ik_dof_pos_arm_noisy += to_tensor_32(
                    self._sample_random(
                        -self._ik_dof_pos_noise_scale,
                        self._ik_dof_pos_noise_scale,
                        size=self._ik_dof_pos_arm_noisy.shape[1],
                    )
                )

            eef_pos, eef_quat = self.kinematics.forward_kinematics(
                self._ik_dof_pos_arm_noisy
            )

            ik_pos_delta = self.target_eef_pose[:, :3] - eef_pos
            ik_orn_delta = torch_utils.orientation_error(
                self.target_eef_pose[:, 3:], eef_quat
            )

            if self.ik_config.ik_version == "v1":
                raise NotImplementedError()
            elif self.ik_config.ik_version == "v2":
                self._ik_target_joint_angles[:], _ = self.kinematics.inverse_kinematics(
                    torch.cat([ik_pos_delta, ik_orn_delta], dim=1),
                    self._ik_dof_pos_arm_noisy,
                    tol=self.ik_config.tol,  # 0.01,
                    max_iters=self.ik_config.max_iters,  # 100,
                    damping=self.ik_config.damping,  # 0.05
                )

            if self.ik_config.joint_target_ema is not None:
                ema = self.ik_config.joint_target_ema
                self._ik_target_joint_angles[:] = (
                    ema * self._prev_ik_target_joint_angles
                    + (1.0 - ema) * self._ik_target_joint_angles
                )

            return np.clip(
                (
                    to_numpy(self._ik_target_joint_angles)
                    - self.default_dof_pos[self.arm_joint_indices]
                )
                / self.config.robot.control.pos_controller_config.action_scale,
                self._act_limit_low[self.arm_joint_indices],
                self._act_limit_high[self.arm_joint_indices],
            )

        if (
            self.config.robot.control.target_delta_pos_controller_config.normalize_action_space
        ):
            action_lim_scale = (
                self.config.robot.control.target_delta_pos_controller_config.action_lim_scale
            )
            action_clip_value = (
                self.config.robot.control.target_delta_pos_controller_config.action_clip_value
            )
            arm_actions = arm_actions / action_lim_scale * action_clip_value

        self.target_dof_pos[:] = np.clip(
            self.target_dof_pos + arm_actions,
            self.dof_pos_limits_low[self.arm_joint_indices],
            self.dof_pos_limits_high[self.arm_joint_indices],
        )

        return np.clip(
            (self.target_dof_pos - self.default_dof_pos[self.arm_joint_indices])
            / self.config.robot.control.pos_controller_config.action_scale,
            self._act_limit_low[self.arm_joint_indices],
            self._act_limit_high[self.arm_joint_indices],
        )

    def step(self, action):
        actions_clipped = np.clip(action, self.act_space_low, self.act_space_high)

        n_finger_joints = len(self.finger_joint_indices)
        hand_actions = actions_clipped[:n_finger_joints]
        arm_actions = actions_clipped[n_finger_joints:]
        self.action[:] = 0.0
        self.action[self.finger_joint_indices] = hand_actions
        with torch.inference_mode():
            self.action[self.arm_joint_indices] = self._rescale_arm_actions(arm_actions)

        # Update target positions based on policy
        action_dof_pos = (
            self.action * self.config.robot.control.pos_controller_config.action_scale
            + self.default_dof_pos
        )

        for _ in range(self.control_decimation):
            # Apply PD control
            tau = pd_control(
                action_dof_pos,
                self.data.qpos[: self.obj_qpos_start],
                self.stiffness,
                np.zeros_like(self.damping),
                self.data.qvel[: self.obj_qvel_start],
                self.damping,
            )
            if self.config.robot.control.clip_torques:
                tau = np.clip(tau, -self.torque_limits, self.torque_limits)
            self.data.ctrl[:] = tau

            # Step simulation
            mujoco.mj_step(self.model, self.data)

        info = self.evaluate()
        self.throw_wait_steps -= 1

        terminated = False
        if not self.config.env.termination.ignore_termination:
            if self.config.env.termination.terminate_by_obj_height:
                terminated = (
                    info["obj_pos"][2]
                    <= self.config.env.termination.scales.termination_obj_height
                )

        if terminated:
            obs, info = self.reset()
        else:
            obs = self._get_obs(info)

        self.elapsed_steps += 1

        return obs, 0, terminated, False, info

    def evaluate(self):
        self.cam_pos = self.data.cam_xpos[self.tracking_camera_id].copy()
        self.cam_mat = self.data.cam_xmat[self.tracking_camera_id].reshape(3, 3).copy()

        robot_qpos = self.data.qpos[:-7].copy()
        robot_qvel = self.data.qvel[:-6].copy()

        hand_pos = self.data.site_xpos[self.hand_site_idx].copy()
        hand_quat = mj2ig_quat(self.data.xquat[self.hand_body_idx].copy())
        hand_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            self.hand_site_idx,
            hand_vel,
            0,
        )
        hand_lin_vel = hand_vel[:3].copy()
        hand_ang_vel = hand_vel[3:].copy()
        hand_goal_pos = tf_apply(hand_quat, hand_pos, self.hand_goal_point_in_hand)

        hand_quat_inv, hand_pos_inv = tf_inverse(hand_quat, hand_pos)

        head_pos = self.data.site_xpos[self.head_site_idx].copy()
        head_quat = mj2ig_quat(
            self.data.xquat[self.model.site_bodyid[self.head_site_idx]].copy()
        )
        head_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            self.head_site_idx,
            head_vel,
            0,
        )
        head_lin_vel = head_vel[:3].copy()
        head_ang_vel = head_vel[3:].copy()

        head_quat_inv, head_pos_inv = tf_inverse(head_quat, head_pos)

        obj_pos = self.data.xpos[self.obj_body_idx].copy()
        obj_quat = mj2ig_quat(self.data.xquat[self.obj_body_idx].copy())
        obj_lin_vel = self.data.qvel[
            self.obj_qvel_start : self.obj_qvel_start + 3
        ].copy()
        obj_ang_vel = self.data.qvel[
            self.obj_qvel_start + 3 : self.obj_qvel_start + 6
        ].copy()
        obj_lin_vel = quat_apply(obj_quat, obj_lin_vel).copy()
        obj_ang_vel = quat_apply(obj_quat, obj_ang_vel).copy()

        if self.throw_wait_steps > 0:
            self.set_obj_init_state()
            obj_pos = self.obj_toss["obj_init_pos"]
            obj_quat = ig2mj_quat(self.obj_toss["obj_init_quat"])
            obj_lin_vel = quat_apply(obj_quat, self.obj_toss["obj_init_lin_vel"]).copy()
            obj_ang_vel = quat_apply(obj_quat, self.obj_toss["obj_init_ang_vel"]).copy()

        if self._tactile_obs:
            self._compute_tactile_forces()

        obj_to_hand_goal_dist = np.linalg.norm(hand_goal_pos - obj_pos)
        obj_near_hand_goal = np.array(
            obj_to_hand_goal_dist
            < self.config.env.projectiles.obj_near_hand_goal_dist_threshold
        )
        obj_static = np.array(
            np.linalg.norm(obj_lin_vel)
            < self.config.env.projectiles.obj_static_threshold
        )
        success = np.array(obj_near_hand_goal & obj_static)

        info = {
            "projectile_types": self.projectile_name,
            "projectile_scales": self.projectile_scale.copy(),
            "projectile_dims_scaled": self.projectile_dims_scaled.copy(),
            "projectile_mass": self.model.body_subtreemass[self.obj_body_idx].copy(),
            #
            "elapsed_steps": self.elapsed_steps,
            #
            "actions": self.action,
            #
            "robot_qpos": robot_qpos,
            "robot_qvel": robot_qvel,
            #
            "obj_near_hand_goal": obj_near_hand_goal,
            "obj_static": obj_static,
            "success": success,
            #
            "hand_pos": hand_pos,
            "hand_quat": hand_quat,
            "hand_lin_vel": hand_lin_vel,
            "hand_ang_vel": hand_ang_vel,
            "hand_goal_pos": hand_goal_pos,
            #
            "hand_pos_inv": hand_pos_inv,
            "hand_quat_inv": hand_quat_inv,
            #
            "head_pos": head_pos,
            "head_quat": head_quat,
            "head_lin_vel": head_lin_vel,
            "head_ang_vel": head_ang_vel,
            #
            "head_pos_inv": head_pos_inv,
            "head_quat_inv": head_quat_inv,
            #
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "obj_lin_vel": obj_lin_vel,
            "obj_ang_vel": obj_ang_vel,
            #
            "obj_target_pos": self.obj_target_pos,
        }
        if self._camera_obs:
            info["cam_latency_idx"] = self.cam_latency_idx

        if self._camera_obs:
            self.cam_pos_queue[1:] = self.cam_pos_queue[:-1].copy()
            self.cam_pos_queue[0] = (
                self._sam2tracker_obj_pos(info, translate_obj_depth=True)
                if self._sam2_obs
                else self._noisy_obj_pos_wrt_cam(info)
            )

        return info

    def _get_obs(self, info):
        obs_list = sorted(self.config.obs.obs_dict.actor_obs)

        obs_dict = {
            k: getattr(self, f"_get_obs_{k}")(info).copy()
            * self.config.obs.obs_scales[k]
            for k in obs_list
        }
        for k, v in obs_dict.items():
            v[:] += (
                self._sample_random(-1.0, 1.0, v.shape)
                * self.config.obs.noise_scales[k]
                * self.config.obs.noise_level
            )
        obs = np.concatenate([obs_dict[k] for k in obs_list])

        if self.history_buffer is not None:
            for k, v in self.history_buffer.items():
                v[1:] = v[:-1]
                v[:1] = obs_dict[k].copy()

        return obs

    def _get_obs_history_actor(self, info):
        assert self.history_buffer is not None
        return np.concatenate(
            [
                self.history_buffer[k].flatten()
                for k in sorted(self.config.obs.obs_auxiliary.history_actor.keys())
            ]
        )

    def _get_obs_dof_pos_finger_arm(self, info):
        return (
            info["robot_qpos"][self.finger_arm_joint_indices]
            - self.default_dof_pos[self.finger_arm_joint_indices]
        )

    def _get_obs_dof_vel_arm(self, info):
        return info["robot_qvel"][self.arm_joint_indices]

    def _get_obs_dof_vel_finger_arm(self, info):
        return info["robot_qvel"][self.finger_arm_joint_indices]

    def _get_obs_actions_finger_arm(self, info):
        return self.action[self.finger_arm_joint_indices]

    def _get_obs_obj_pose_wrt_hand(self, info):
        hand_quat_inv, hand_pos_inv = info["hand_quat_inv"], info["hand_pos_inv"]
        obj_pos_in_hand = tf_apply(hand_quat_inv, hand_pos_inv, info["obj_pos"])
        obj_quat_in_hand = quat_mul(hand_quat_inv, info["obj_quat"])
        return np.concatenate([obj_pos_in_hand, obj_quat_in_hand])

    def _get_obs_obj_target_pos_wrt_hand(self, info):
        hand_quat_inv, hand_pos_inv = info["hand_quat_inv"], info["hand_pos_inv"]
        obj_target_pos_wrt_hand = tf_apply(
            hand_quat_inv, hand_pos_inv, info["obj_target_pos"]
        )
        return obj_target_pos_wrt_hand

    def _get_obs_obj_lin_vel_wrt_hand(self, info):
        return quat_rotate_inverse(
            info["hand_quat"], info["obj_lin_vel"] - info["hand_lin_vel"]
        )

    def _get_obs_obj_ang_vel_wrt_hand(self, info):
        return quat_rotate_inverse(
            info["hand_quat"], info["obj_ang_vel"] - info["hand_ang_vel"]
        )

    def _get_obs_obj_pose_wrt_head(self, info):
        head_quat_inv, head_pos_inv = info["head_quat_inv"], info["head_pos_inv"]
        obj_pos_in_head = tf_apply(head_quat_inv, head_pos_inv, info["obj_pos"])
        obj_quat_in_head = quat_mul(head_quat_inv, info["obj_quat"])
        return np.concatenate([obj_pos_in_head, obj_quat_in_head])

    def _get_obs_obj_pos_wrt_head(self, info):
        head_quat_inv, head_pos_inv = info["head_quat_inv"], info["head_pos_inv"]
        obj_pos_in_head = tf_apply(head_quat_inv, head_pos_inv, info["obj_pos"])
        return obj_pos_in_head

    def _get_obs_obj_target_pos_wrt_head(self, info):
        head_quat_inv, head_pos_inv = info["head_quat_inv"], info["head_pos_inv"]
        obj_target_pos_wrt_head = tf_apply(
            head_quat_inv, head_pos_inv, info["obj_target_pos"]
        )
        return obj_target_pos_wrt_head

    def _get_obs_obj_lin_vel_wrt_head(self, info):
        return quat_rotate_inverse(
            info["head_quat"], info["obj_lin_vel"] - info["head_lin_vel"]
        )

    def _get_obs_obj_ang_vel_wrt_head(self, info):
        return quat_rotate_inverse(
            info["head_quat"], info["obj_ang_vel"] - info["head_ang_vel"]
        )

    def _get_obs_obj_pos_wrt_camera_noisy(self, info):
        assert self._camera_obs and not self._sam2_obs
        return self.cam_pos_queue[self.cam_latency_idx]

    def _get_obs_obj_pos_wrt_camera_noisy_sam2(self, info):
        assert self._sam2_obs
        pos = self.cam_pos_queue[self.cam_latency_idx]
        x_cam, y_cam, z_cam = -pos[1], -pos[2], (pos[0] + 1e-9)
        u = int(
            self.cam_config.fx * (x_cam / z_cam) + self.cam_config.cx,
        )
        v = int(
            self.cam_config.fy * (y_cam / z_cam) + self.cam_config.cy,
        )
        rad = 5
        self.tracking_rgb[v - rad : v + rad, u - rad : u + rad, 0] = 0.0
        self.tracking_rgb[v - rad : v + rad, u - rad : u + rad, 1] = 255.0
        self.tracking_rgb[v - rad : v + rad, u - rad : u + rad, 2] = 255.0
        return pos

    def _noisy_obj_pos_wrt_cam(self, info):
        """Isaacgym camera noise model: region noise, bias, frame mask, drops."""
        cam = self.cam_config
        # mujoco camera frame is x-right, y-up, z-back
        # isaacgym camera frame is x-forward, y-left, z-up
        obj_pos_mj_cam = self.cam_mat.T @ (info["obj_pos"] - self.cam_pos)
        obj_pos_cam_clean = np.array(
            [-obj_pos_mj_cam[2], -obj_pos_mj_cam[0], obj_pos_mj_cam[1]]
        )
        x_cam, y_cam, z_cam = (
            -obj_pos_cam_clean[1],
            -obj_pos_cam_clean[2],
            obj_pos_cam_clean[0],
        )
        obj_in_frame = False
        obj_near_cam_edge = False
        if z_cam > 0.0:
            u = cam.fx * (x_cam / z_cam) + cam.cx
            v = cam.fy * (y_cam / z_cam) + cam.cy
            obj_in_frame = 0.0 <= u < cam.width and 0.0 <= v < cam.height
            obj_near_cam_edge = obj_in_frame and (
                u < cam.edge_margin_px
                or u >= cam.width - cam.edge_margin_px
                or v < cam.edge_margin_px
                or v >= cam.height - cam.edge_margin_px
            )
        obj_near_hand = np.linalg.norm(info["obj_pos"] - info["hand_pos"]) < 0.2
        if obj_near_hand:
            base_sigma = cam.noise_near_hand
        elif obj_near_cam_edge:
            base_sigma = cam.noise_edge
        else:
            base_sigma = cam.noise_clear
        # isaacgym zeroes object velocity while waiting for the throw
        obj_speed = (
            0.0 if self.throw_wait_steps > 0 else np.linalg.norm(info["obj_lin_vel"])
        )
        sigma = (
            base_sigma
            + np.linalg.norm(info["obj_pos"] - self.cam_pos) * cam.noise_distance_factor
            + obj_speed * cam.noise_velocity_factor
        )
        obj_pos_cam_noisy = (
            obj_pos_cam_clean + self._sample_random(-1.0, 1.0, 3) * sigma
        )
        if self.config.domain_rand.randomize_cam_pos_bias:
            obj_pos_cam_noisy += self.cam_pos_bias
        if not obj_in_frame:
            return np.zeros(3)
        if self._sample_random(0.0, 1.0) < cam.no_pred_prob:
            return np.zeros(3)
        return obj_pos_cam_noisy

    def _sam2tracker_obj_pos(self, info, translate_obj_depth=False):
        assert self.obj_tracker is not None

        self.tracking_renderer.update_scene(self.data, camera=self.tracking_camera_id)
        rgb = self.tracking_renderer.render()
        self.tracking_renderer.enable_depth_rendering()
        depth = self.tracking_renderer.render()
        self.tracking_renderer.disable_depth_rendering()

        # depth noise, we only add clear noise in the form of depth reading noise
        depth += self._sample_random(
            -self.config.camera.noise_clear,
            self.config.camera.noise_clear,
            depth.shape,
        )

        self.tracking_rgb = rgb

        if not self.obj_tracker.initialized:
            obj_pos_wrt_cam = self.cam_mat.T @ (info["obj_pos"] - self.cam_pos)
            x_cam, y_cam, z_cam = (
                obj_pos_wrt_cam[0],
                -obj_pos_wrt_cam[1],
                -obj_pos_wrt_cam[2],
            )
            u = round(self.cam_config.fx * (x_cam / z_cam) + self.cam_config.cx)
            v = round(self.cam_config.fy * (y_cam / z_cam) + self.cam_config.cy)
            pos = self.obj_tracker.init_predictor(
                rgb,
                depth,
                (u, v),
                (
                    self._get_obs_obj_dims_scaled(info).mean() / 2
                    if translate_obj_depth
                    else 0.0
                ),
            )
            assert pos is not None
        else:
            pos = self.obj_tracker.predict_pos(
                rgb,
                depth,
                (
                    self._get_obs_obj_dims_scaled(info).mean() / 2
                    if translate_obj_depth
                    else 0.0
                ),
            )

        # ig2cam
        # x_cam = -self._obj_pos_wrt_cam_noisy[:, 1].clone()
        # y_cam = -self._obj_pos_wrt_cam_noisy[:, 2].clone()
        # z_cam = self._obj_pos_wrt_cam_noisy[:, 0].clone()
        # x = -y; y = -z; z = x

        if pos is None:
            return np.zeros(3)

        # cam2ig
        return np.array([pos[2], -pos[0], -pos[1]])

    def _get_obs_obj_scale(self, info):
        return self.projectile_scale

    def _get_obs_obj_dims(self, info):
        return self.projectile_dims

    def _get_obs_obj_dims_scaled(self, info):
        return self.projectile_dims_scaled

    def _get_obs_tactile_forces_short(self, info):
        assert self._tactile_obs
        """Tactile forces from fingers 0, 1, 2"""
        return self.tactile_forces[0:3].copy()

    def _get_obs_occlusion_flag(self, info):
        assert self._tactile_obs
        """Binary flag indicating if object is occluded (1.0) or visible (0.0)"""
        return self.obj_occluded_flag.copy()

    def _get_obs_tactile_palm(self, info):
        assert self._tactile_obs
        """Tactile forces from palm only"""
        return self.tactile_forces[5:6].copy()

    def render(self):
        if self.render_mode == "rgb_array":
            if self.tracking_rgb is not None:
                return self.tracking_rgb
            if self.tracking_renderer is None:
                self._init_renderer()
            self.tracking_renderer.update_scene(
                self.data, camera=self.tracking_camera_id
            )
            return self.tracking_renderer.render()
        elif self.render_mode == "human":
            if not self.renderer.is_running():
                self.close()
            self.renderer.sync()
            return None

        return None

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

        if self.tracking_renderer is not None:
            self.tracking_renderer.close()
