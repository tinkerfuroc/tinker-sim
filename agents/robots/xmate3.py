from pathlib import Path

import numpy as np
import sapien.core as sapien
from challenge_grasp import DESCRIPTION_DIR
from challenge_grasp.agents.base_agent import BaseAgent, parse_urdf_config
from challenge_grasp.agents.configs.xmate3 import defaults
from challenge_grasp.utils.common import compute_angle_between
from challenge_grasp.utils.sapien_utils import (
    get_entity_by_name,
    get_pairwise_contact_impulse,
)


class Xmate3Robotiq(BaseAgent):
    _config: defaults.Xmate3RobotiqDefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.Xmate3RobotiqDefaultConfig()

    def _after_init(self):
        self.finger1_link: sapien.LinkBase = get_entity_by_name(
            self.robot.get_links(), "robotiq_140_left_inner_finger_pad"
        )
        self.finger2_link: sapien.LinkBase = get_entity_by_name(
            self.robot.get_links(), "robotiq_140_right_inner_finger_pad"
        )

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[:3, 2]
        rdirection = self.finger2_link.pose.to_transformation_matrix()[:3, 2]

        # angle between impulse and open direction
        langle = compute_angle_between(ldirection, limpulse)
        rangle = compute_angle_between(rdirection, rimpulse)

        lflag = (
            np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
        )
        rflag = (
            np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle
        )

        return all([lflag, rflag])

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(approaching, closing)
        T = np.eye(4)
        T[:3, :3] = np.stack([approaching, closing, ortho], axis=1)
        T[:3, 3] = center
        return sapien.Pose.from_transformation_matrix(T)

    def _load_articulation(self):
        loader = self.scene.create_urdf_loader()

        urdf_path = str(self.urdf_path)
        urdf_path = urdf_path.format(description=DESCRIPTION_DIR)
        urdf_config = parse_urdf_config(self.urdf_config, self.scene)

        builder = loader.load_file_as_articulation_builder(urdf_path, urdf_config)

        # Disable self collision for simplification
        for link_builder in builder.get_link_builders():
            link_builder.set_collision_groups(1, 1, 2, 0)
        self.robot = builder.build(fix_root_link=self.fix_root_link)
        assert self.robot is not None, f"Fail to load URDF from {urdf_path}"
        self.robot.set_name(Path(urdf_path).stem)

    def _add_constraints(self):
        # Add constraints
        outer_knuckle = next(
            j
            for j in self.robot.get_active_joints()
            if j.name == "right_outer_knuckle_joint"
        )
        outer_finger = next(
            j
            for j in self.robot.get_active_joints()
            if j.name == "right_inner_finger_joint"
        )
        inner_knuckle = next(
            j
            for j in self.robot.get_active_joints()
            if j.name == "right_inner_knuckle_joint"
        )

        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        T_pw = pad.pose.inv().to_transformation_matrix()
        p_w = (
            outer_finger.get_global_pose().p
            + inner_knuckle.get_global_pose().p
            - outer_knuckle.get_global_pose().p
        )
        T_fw = lif.pose.inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]

        right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f), pad, sapien.Pose(p_p)
        )
        right_drive.lock_motion(1, 1, 1, 0, 0, 0)

        outer_knuckle = next(
            j
            for j in self.robot.get_active_joints()
            if j.name == "left_outer_knuckle_joint"
        )
        outer_finger = next(
            j
            for j in self.robot.get_active_joints()
            if j.name == "left_inner_finger_joint"
        )
        inner_knuckle = next(
            j
            for j in self.robot.get_active_joints()
            if j.name == "left_inner_knuckle_joint"
        )

        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        T_pw = pad.pose.inv().to_transformation_matrix()
        p_w = (
            outer_finger.get_global_pose().p
            + inner_knuckle.get_global_pose().p
            - outer_knuckle.get_global_pose().p
        )
        T_fw = lif.pose.inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]

        left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f), pad, sapien.Pose(p_p)
        )
        left_drive.lock_motion(1, 1, 1, 0, 0, 0)

        left_link = next(
            l
            for l in self.robot.get_links()
            if l.name == "robotiq_140_left_outer_knuckle"
        )
        right_link = next(
            l
            for l in self.robot.get_links()
            if l.name == "robotiq_140_right_outer_knuckle"
        )
        self.scene.create_gear(left_link, sapien.Pose(), right_link, sapien.Pose())

        # Cache robot link ids
        self.robot_link_ids = [link.get_id() for link in self.robot.get_links()]
