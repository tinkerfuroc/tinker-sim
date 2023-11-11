from challenge_grasp import DESCRIPTION_DIR
from challenge_grasp.agents.controllers import *
from challenge_grasp.sensors.camera import CameraConfig
from challenge_grasp.sensors.depth_camera import StereoDepthCameraConfig


class UR5eRobotiq85DefaultConfig:
    def __init__(self) -> None:
        self.urdf_path = f"{DESCRIPTION_DIR}/ur5e/ur5e_robotiq85.urdf"
        self.urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            # link=dict(
            #     left_inner_finger_pad=dict(
            #         material="gripper", patch_radius=0.1, min_patch_radius=0.1
            #     ),
            #     right_inner_finger_pad=dict(
            #         material="gripper", patch_radius=0.1, min_patch_radius=0.1
            #     ),
            # ),
        )

        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.arm_stiffness = 3000  # 1e3
        self.arm_damping = 100  # 1e2
        self.arm_force_limit = 100
        self.arm_delta = 0.05

        # self.gripper_joint_names = [
        #     "robotiq_85_left_knuckle_joint",
        #     "robotiq_85_right_knuckle_joint",
        #     "robotiq_85_left_inner_knuckle_joint",
        #     "robotiq_85_right_inner_knuckle_joint",
        # ]
        self.gripper_joint_names = [
            "robotiq_85_left_knuckle_joint",
            "robotiq_85_right_knuckle_joint",
            "robotiq_85_left_inner_knuckle_joint",
            "robotiq_85_right_inner_knuckle_joint",
            "robotiq_85_left_finger_tip_joint",
            "robotiq_85_right_finger_tip_joint"
        ]
        self.gripper_stiffness = 1000  # 1e3
        self.gripper_damping = 100  # 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "grasp_convenient_link"
        # self.ee_delta = 0.05
        self.ee_delta = 0.03
        self.rot_bound = 0.1
        self.rot_euler_bound = 0.05  # ~2.86deg

    @property
    def controllers(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -self.arm_delta,
            self.arm_delta,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            self.arm_joint_names,
            -self.ee_delta,
            self.ee_delta,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -self.ee_delta,
            self.ee_delta,
            self.rot_bound,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
        )
        arm_pd_ee_delta_pose_euler = PDEEPoseEulerControllerConfig(
            self.arm_joint_names,
            -self.ee_delta,
            self.ee_delta,
            self.rot_euler_bound,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0,
            0.8,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
        )

        controller_configs = dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_ee_delta_pose_euler=dict(
                arm=arm_pd_ee_delta_pose_euler, gripper=gripper_pd_joint_pos
            )
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        return [
            # CameraConfig(
            #     uid="base_camera",
            #     p=[0.0, 0.0, 0.0],
            #     q=[1, 0, 0, 0],
            #     width=128,
            #     height=128,
            #     fov=1.5707,
            #     near=0.01,
            #     far=10,
            #     actor_uid="camera_base_link",
            #     hide_link=False,
            # ),
            # CameraConfig(
            #     uid="hand_camera",
            #     p=[0.0, 0.0, 0.0],
            #     q=[1, 0, 0, 0],
            #     width=128,
            #     height=128,
            #     fov=1.5707,
            #     near=0.01,
            #     far=10,
            #     actor_uid="camera_hand_link",
            #     hide_link=False,
            # ),  # sapien camera config
            # CameraConfig(
            #     uid="hand_camera",
            #     p=[0.0, 0.0, 0.0],
            #     q=[1, 0, 0, 0],
            #     width=640,
            #     height=360,
            #     fov=0.7407,
            #     near=0.2,
            #     far=2,
            #     actor_uid="camera_hand_link",
            #     hide_link=False,
            # ),  # sapien camera config
            CameraConfig(
                uid="hand_camera",
                p=[0.0, 0.0, 0.0],
                q=[1, 0, 0, 0],
                width=128,
                height=128,
                fov=1.5707,
                near=0.01,
                far=10,
                actor_uid="camera_hand_link",
                hide_link=False,
            ),  # sapien camera config
            CameraConfig(
                uid="hand_realsense",
                p=[0.0, 0.0, 0.0],
                q=[1, 0, 0, 0],
                width=640,
                height=360,
                fov=0.7407,
                near=0.01,
                far=2,
                actor_uid="camera_hand_link",
                hide_link=False,
            ),  # sapien camera config
            CameraConfig(
                uid="base_kinect",
                # p=[0.86, 0.0, 0.6],
                p=[1, 0, 1],
                q=[0, 0.433, 0, -0.9],
                width=640,
                height=360,
                fov=0.7407,
                near=0.01,
                far=2,
                # actor_uid="camera_hand_link",
                # hide_link=False,
            ),  # sapien camera config
            # StereoDepthCameraConfig(
            #     uid="hand_camera",
            #     p=[0.0, 0.0, 0.0],
            #     q=[1, 0, 0, 0],
            #     width=1280,
            #     height=720,
            #     fov=0.7407,
            #     near=0.2,
            #     far=2,
            #     actor_uid="camera_hand_link",
            #     hide_link=False,
            #     texture_names=["Color", "Position", "Segmentation"],
            #     # texture_names=["Color"],
            # )
        ]
