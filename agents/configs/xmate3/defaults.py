from challenge_grasp import DESCRIPTION_DIR
from challenge_grasp.agents.controllers import *
from challenge_grasp.sensors.camera import CameraConfig


class Xmate3RobotiqDefaultConfig:
    def __init__(self) -> None:
        self.urdf_path = f"{DESCRIPTION_DIR}/xmate3_robotiq/xmate3_robotiq.urdf"
        self.urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                left_inner_finger_pad=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                right_inner_finger_pad=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),
        )

        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100
        self.arm_delta = 0.1

        self.gripper_joint_names = [
            "left_outer_knuckle_joint",
            "right_outer_knuckle_joint",
        ]
        self.gripper_stiffness = 3000  # 3000
        self.gripper_damping = 30  # 30
        self.gripper_force_limit = 400  # 400

        self.ee_link_name = "grasp_convenient_link"
        self.ee_delta = 0.1

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
            self.ee_delta,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0,
            0.7,
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
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        """
        Intrinsic of "Color" / 424x240 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y16}
        Width:      	424
        Height:     	240
        PPX:        	210.168258666992
        PPY:        	117.593872070312
        Fx:         	303.756774902344
        Fy:         	303.421295166016
        Distortion: 	Inverse Brown Conrady
        Coeffs:     	0  	0  	0  	0  	0
        FOV (deg):  	69.82 x 43.16
        """
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
            # ),
            CameraConfig(
                uid="hand_camera",
                p=[0.0, 0.0, 0.0],
                q=[1, 0, 0, 0],
                width=1280,
                height=720,
                fov=0.7407,
                near=0.2,
                far=2,
                actor_uid="camera_hand_link",
                hide_link=False,
            ),  # sapien camera config
        ]
