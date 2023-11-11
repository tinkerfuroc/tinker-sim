from pathlib import Path

import numpy as np
import sapien.core as sapien
from challenge_grasp import DESCRIPTION_DIR
from challenge_grasp.agents.base_agent import BaseAgent, parse_urdf_config
from challenge_grasp.agents.configs.ur5e_robotiq140_old import defaults
from challenge_grasp.utils.common import compute_angle_between
from challenge_grasp.utils.sapien_utils import (
    get_entity_by_name,
    get_pairwise_contact_impulse,
    get_multi_pairwise_contact_impulse,
)
from challenge_grasp.utils.trimesh_utils import get_actor_mesh


class UR5e_Robotiq140_old(BaseAgent):
    _config: defaults.UR5eRobotiq140oldDefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.UR5eRobotiq140oldDefaultConfig()

    def _after_init(self):
        self.finger1_link: sapien.LinkBase = get_entity_by_name(
            self.robot.get_links(), "left_inner_finger_pad"
        )
        self.finger2_link: sapien.LinkBase = get_entity_by_name(
            self.robot.get_links(), "right_inner_finger_pad"
        )
        self.finger1_mesh = get_actor_mesh(self.finger1_link, False)
        self.finger2_mesh = get_actor_mesh(self.finger2_link, False)
        self.finger_size = (0.03, 0.07, 0.0075)  # values from URDF

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

    def check_contact(self, actor: sapien.ActorBase, min_impulse=1e-6):
        contacts = self.scene.get_contacts()
        # obj_contact_actors = [contact.actor1 for contact in contacts if contact.actor0 == actor] \
        #                      + [contact.actor0 for contact in contacts if contact.actor1 == actor]
        # contact_actors = [c_actor for c_actor in obj_contact_actors if c_actor in self.robot_collision_actors]
        multi_impluse = np.linalg.norm(
            get_multi_pairwise_contact_impulse(contacts, self.robot_collision_actors, actor), axis=-1
        )
        is_agent_contact = any(multi_impluse >= min_impulse)
        return is_agent_contact, multi_impluse

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
        # Cache robot link ids
        self.robot_link_ids = [link.get_id() for link in self.robot.get_links()]
        self.robot_collision_actors = [actor for actor in self.robot.get_links() if not actor.get_collision_shapes() == []]
        active_joint_name = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.gripper_joint_ids = [active_joint_name.index(joint_name) for joint_name in self.config.gripper_joint_names]
