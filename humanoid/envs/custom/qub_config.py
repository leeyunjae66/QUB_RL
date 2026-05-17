# SPDX-License-Identifier: BSD-3-Clause

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class QUBCfg(LeggedRobotCfg):
    """Configuration for the QUB humanoid URDF."""

    class env(LeggedRobotCfg.env):
        frame_stack = 15
        c_frame_stack = 3
        num_actions = 13
        num_single_obs = 11 + 3 * num_actions
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 25 + 4 * num_actions
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_envs = 4096
        episode_length_s = 24
        use_ref_actions = False

    class safety:
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 1.5

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/QUB/urdf/QUB.urdf'
        name = "QUB"

        # Fixed foot links are collapsed by Isaac Gym, so use the ankle roll bodies.
        foot_name = "Ankle_roll"
        knee_name = "Thigh_pitch_3"

        terminate_after_contacts_on = []
        penalize_contacts_on = ["base_link"]
        self_collisions = 1
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20
        num_cols = 20
        max_init_terrain_level = 10
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.95]

        default_joint_angles = {
            'Torso_yaw_joint': 0.,
            'R_Hip_pitch_joint': 0.,
            'R_Hip_roll_joint': 0.,
            'R_Hip_yaw_joint': 0.,
            'R_Thigh_pitch_3_joint': 0.,
            'R_Ankle_pitch_joint': 0.,
            'R_Ankle_roll_joint': 0.,
            'L_Hip_pitch_joint': 0.,
            'L_Hip_roll_joint': 0.,
            'L_Hip_yaw_joint': 0.,
            'L_Thigh_pitch_3_joint': 0.,
            'L_Ankle_pitch_joint': 0.,
            'L_Ankle_roll_joint': 0.,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {
            'Torso_yaw': 80.0,
            'Hip_pitch': 200.0,
            'Hip_roll': 160.0,
            'Hip_yaw': 120.0,
            'Thigh_pitch': 220.0,
            'Ankle_pitch': 40.0,
            'Ankle_roll': 30.0,
        }
        damping = {
            'Torso_yaw': 5.0,
            'Hip_pitch': 8.0,
            'Hip_roll': 6.0,
            'Hip_yaw': 5.0,
            'Thigh_pitch': 8.0,
            'Ankle_pitch': 4.0,
            'Ankle_roll': 3.0,
        }
        action_scale = 0.25
        decimation = 10

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        up_axis = 1

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.1
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class domain_rand:
        randomize_friction = False
        friction_range = [0.1, 2.0]
        randomize_base_mass = False
        added_mass_range = [-2., 2.]
        push_robots = False
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        action_delay = 0.0
        action_noise = 0.0

    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 6.
        heading_command = False

        class ranges:
            lin_vel_x = [0.25, 0.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards:
        base_height_target = 0.85
        base_height_termination = 0.60
        min_dist = 0.12
        max_dist = 0.38
        target_joint_pos_scale = 0.12
        target_feet_height = 0.04
        cycle_time = 0.7
        only_positive_rewards = True
        tracking_sigma = 5
        max_contact_force = 350

        class scales:
            joint_pos = 0.4
            feet_clearance = 0.5
            feet_contact_number = 0.8
            feet_air_time = 0.5
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.1
            feet_contact_forces = -0.01
            tracking_lin_vel = 2.5
            tracking_ang_vel = 0.8
            vel_mismatch_exp = 0.5
            low_speed = 0.8
            track_vel_hard = 1.0
            default_joint_pos = 0.2
            orientation = 0.8
            base_height = 0.2
            base_acc = 0.15
            action_smoothness = -0.004
            torques = -2e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.

    class viewer(LeggedRobotCfg.viewer):
        pos = [3., -3., 2.]
        lookat = [0., 0., 0.6]


class QUBCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005
        learning_rate = 3e-4
        num_learning_epochs = 5
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60
        max_iterations = 3001
        save_interval = 100
        experiment_name = 'QUB_ppo'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
