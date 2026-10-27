import xml.etree.ElementTree as ET

from line_profiler import profile

import numpy as np
import torch
import torch.autograd.functional as autogradF
import torch.nn as nn

from humanoid.envs.humanoidverse.utils.torch_utils import *
from humanoid.envs.humanoidverse.utils.torch_utils import get_euler_xyz


class Kinematics(nn.Module):
    def __init__(
        self,
        urdf_path,
        base_link_name,
        end_effector_link_name,
        joint_indices,
        valid_search_links,
        device,
    ):
        super().__init__()

        self.base_link_name = base_link_name
        self.end_effector_link_name = end_effector_link_name
        self.joint_indices = joint_indices
        self.valid_search_links = valid_search_links
        self.device = device
        self.robot = self._parse_urdf(urdf_path)

        self.I3 = torch.eye(3, device=self.device).reshape(1, 3, 3)

    @profile
    def forward_kinematics(self, joint_angles):
        end_effector_pose = self._compute_pose(
            joint_angles, self.base_link_name
        )  # [B, 4, 4]
        quat = self._pose_to_quat(end_effector_pose)

        return end_effector_pose[:, 0:3, 3], quat

    def forward_kinematics_euler(self, joint_angles):
        eef_pos, eef_quat = self.forward_kinematics(joint_angles)
        eef_euler = get_euler_xyz(eef_quat)
        eef_euler = torch.stack(eef_euler, dim=-1)

        return torch.cat((eef_pos, eef_euler), dim=-1)

    @profile
    def _pose_to_quat(self, pose, eps=1e-16):
        batch_size = pose.shape[0]
        quat = torch.zeros((batch_size, 4), device=pose.device)
        quat[:, 3] = 1 + pose[:, 0, 0] + pose[:, 1, 1] + pose[:, 2, 2]

        quat[:, 3] = torch.clamp(quat[:, 3], min=eps)
        quat[:, 3] = torch.sqrt(quat[:, 3]) / 2

        quat[:, 0] = (pose[:, 2, 1] - pose[:, 1, 2]) / (4 * quat[:, 3])
        quat[:, 1] = (pose[:, 0, 2] - pose[:, 2, 0]) / (4 * quat[:, 3])
        quat[:, 2] = (pose[:, 1, 0] - pose[:, 0, 1]) / (4 * quat[:, 3])

        return quat

    def _parse_urdf(self, urdf_path):
        robot = {"links": {}, "joints": {}}

        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for link in root.findall("link"):
            link_data = {
                "name": link.get("name"),
                "parent": None,
                "children": [],
                "pose": self._parse_origin(link.find("visual/origin")),
            }
            robot["links"][link_data["name"]] = link_data

        for joint in root.findall("joint"):
            axis = joint.find("axis")

            if axis is not None:
                axis = np.fromstring(axis.get("xyz"), sep=" ")
                axis = torch.from_numpy(axis).to(self.device).float()

                axis_unit = axis / torch.linalg.norm(axis)
                ax, ay, az = axis_unit  # scalars (0-d tensors)
                # skew-symmetric cross-product matrix K for axis
                joint_K = torch.tensor(
                    [[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]], device=self.device
                ).view(1, 3, 3)
                joint_aaT = axis_unit.view(3, 1) @ axis_unit.view(1, 3)  # [3,3]
            else:
                axis = None
                axis_unit = None
                joint_K = None
                joint_aaT = None

            joint_data = {
                "name": joint.get("name"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "type": joint.get("type"),
                "axis": axis,
                "axis_unit": axis_unit,
                "joint_K": joint_K,
                "joint_aaT": joint_aaT,
                "pose": self._parse_origin(joint.find("origin")),
            }
            robot["joints"][joint_data["name"]] = joint_data

            robot["links"][joint_data["parent"]]["children"].append(joint_data["child"])

        return robot

    def _parse_origin(self, origin_element):
        if origin_element is None:
            return torch.eye(4, device=self.device)
        else:
            xyz = np.fromstring(origin_element.get("xyz", "0 0 0"), sep=" ")
            xyz = torch.from_numpy(xyz).to(self.device).float()
            rpy = np.fromstring(origin_element.get("rpy", "0 0 0"), sep=" ")
            rpy = torch.from_numpy(rpy).to(self.device).float()
            return self._pose_from_xyz_rpy(xyz, rpy)

    def _pose_from_xyz_rpy(self, xyz, rpy):
        pose = torch.eye(4, device=self.device)
        pose[0:3, 3] = xyz
        roll, pitch, yaw = rpy
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        rot_x = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], device=self.device)
        rot_y = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], device=self.device)
        rot_z = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], device=self.device)
        pose[0:3, 0:3] = rot_z @ rot_y @ rot_x
        return pose

    @profile
    def _compute_pose(self, joint_angles, link_name):
        batch_size = joint_angles.shape[0]
        link_data = self.robot["links"][link_name]
        pose = link_data["pose"].repeat((batch_size, 1, 1))

        if link_name == self.end_effector_link_name:
            return pose

        for child_link_name in link_data["children"]:
            # NOTE (arth): bad if dummy joints in between, but it's ok for g1 arm
            if child_link_name not in self.valid_search_links:
                continue
            joint_name = None
            for candidate_joint_name, joint_data in self.robot["joints"].items():
                if (
                    joint_data["parent"] == link_name
                    and joint_data["child"] == child_link_name
                ):
                    joint_name = candidate_joint_name
                    break

            if joint_name is None:
                raise ValueError(
                    f"Failed to find joint connecting {link_name} and {child_link_name}"
                )

            if joint_name in self.joint_indices:
                joint_angle = joint_angles[:, self.joint_indices[joint_name]]
            else:
                joint_angle = joint_angles[:, 0] * 0

            ee_pose = self._compute_pose(joint_angles, child_link_name)
            joint_transform = self._joint_transform(
                joint_angle,
                joint_data["joint_K"],
                joint_data["joint_aaT"],
            )
            result = torch.bmm(joint_transform, ee_pose)
            result = torch.bmm(joint_data["pose"].repeat((batch_size, 1, 1)), result)
            result = torch.bmm(pose, result)

        return result

    @profile
    def _joint_transform(self, angle, joint_K, joint_aaT):
        B = angle.shape[0]

        if joint_K is None:
            return torch.eye(4).to(self.device).repeat((B, 1, 1))

        ct = torch.cos(angle).view(B, 1, 1)
        st = torch.sin(angle).view(B, 1, 1)
        vt = 1.0 - ct

        R = ct * self.I3 + vt * joint_aaT.view(1, 3, 3) + st * joint_K  # [B,3,3]

        T = torch.eye(4, device=self.device).expand(B, 4, 4).clone()
        T[:, :3, :3] = R
        return T

    def _compute_jac_autograd(self, joint_angles):
        # import pdb; pdb.set_trace()
        # Compute the jacobian using autograd
        result = []

        for angles in joint_angles:
            jac = autogradF.jacobian(self.forward_kinematics_euler, angles[None, :])
            result.append(jac)

        return torch.stack(result)

    @profile
    def _compute_jac(self, joint_angles, eps=1e-2):
        B, N = joint_angles.shape

        eef_pos, eef_quat = self.forward_kinematics(joint_angles)

        # Build all perturbations in one batch: shape [B*N, N]
        angles_rep = joint_angles.repeat_interleave(N, dim=0)
        row = torch.arange(B * N, device=joint_angles.device)
        col = torch.arange(N, device=joint_angles.device).repeat(B)
        angles_rep[row, col] += eps

        pos_p, quat_p = self.forward_kinematics(angles_rep)
        pos_p = pos_p.view(B, N, 3)
        quat_p = quat_p.view(B, N, 4)

        linear = (pos_p - eef_pos[:, None, :]) / eps

        # quat_diff = quat_p * conj(eef_quat)
        quat_base = eef_quat[:, None, :].expand(B, N, 4).reshape(B * N, 4)
        quat_diff = quat_mul(quat_p.reshape(B * N, 4), quat_conjugate(quat_base)).view(
            B, N, 4
        )

        angular = (quat_diff[:, :, 0:3] * torch.sign(quat_diff[:, :, 3:4])) / eps * 2.0

        jac = torch.cat([linear, angular], dim=2)  # [B, N, 6]
        jac = jac.permute(0, 2, 1)  # [B, 6, N]

        # all values are not nan
        assert not torch.isnan(jac).any()

        return jac

    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    @profile
    def inverse_kinematics(
        self, dpose, joint_angles, tol=1e-3, only_pos=False, max_iters=1, damping=0.05
    ):
        num_iters = 0
        batch_size = dpose.shape[0]
        error = np.inf * torch.ones((batch_size), device=self.device)

        end_effector_pos, end_effector_quat = self.forward_kinematics(joint_angles)
        desired_pos = end_effector_pos + dpose[:, :3]
        quat_delta = torch.zeros((batch_size, 4), device=self.device)
        quat_delta[:, :3] = dpose[:, 3:]
        quat_delta[:, 3] = 1
        quat_delta = quat_delta / torch.norm(quat_delta, dim=1, keepdim=True)
        desired_rot = quat_mul(quat_delta, end_effector_quat)
        desired_pose = torch.cat((desired_pos, desired_rot), dim=1)

        while num_iters < max_iters and torch.any(error > tol):
            end_effector_pos, end_effector_quat = self.forward_kinematics(joint_angles)
            pos_delta = desired_pose[:, :3] - end_effector_pos
            orn_delta = self.orientation_error(desired_pose[:, 3:], end_effector_quat)
            dpose = torch.cat((pos_delta, orn_delta), dim=1)

            j_eef = self._compute_jac(joint_angles)
            j_eef_T = torch.transpose(j_eef, 1, 2)
            lmbda = torch.eye(6, device=self.device) * (damping**2)

            if only_pos:
                u = (
                    (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda))[:, :, :3]
                    @ dpose[:, :3].unsqueeze(-1)
                ).squeeze()
            else:
                u = (
                    j_eef_T
                    @ torch.inverse(j_eef @ j_eef_T + lmbda)
                    @ dpose.unsqueeze(-1)
                ).squeeze()

            error = u
            joint_angles = joint_angles + u
            num_iters += 1

        return joint_angles, error

    # def _compute_jacobian(self, joint_angles):
    #     num_joints = len(joint_angles)
    #     end_effector_pose, jacobian_columns = self._compute_pose_and_jacobian_columns(joint_angles, self.end_effector_link_name)
    #     jacobian = np.zeros((3, num_joints))

    #     for i, column in enumerate(jacobian_columns):
    #         jacobian[:, i] = np.dot(column[0:3, 0:3], column[0:3, 3])

    #     return jacobian
