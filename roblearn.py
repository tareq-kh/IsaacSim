from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole
from omniisaacgymenvs.robots.articulations.jetbot import Jetbot

from omni.isaac.core.utils.nucleus import get_assets_root_path

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math

class RoblearnTask(RLTask):
    def __init__(
        self,
        name,                # name of the Task
        sim_config,    # SimConfig instance for parsing cfg
        env,          # env instance of VecEnvBase or inherited class
        offset=None               # transform offset in World
    ) -> None:
         
        # parse configurations, set task-specific members
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._jetbot_positions = torch.tensor([0.0, 0.0, 0.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        #self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._num_observations = 16
        self._num_actions = 2

        
        self._goal_position = [10.0, 0.0, 3.1]
        self._max_velocity = 10.0

        RLTask.__init__(self, name, env)
        return
    
    def get_jetbot(self):
        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        jetbot = Jetbot(prim_path=self.default_zero_env_path + "/Jetbot" , name="Jetbot",usd_path=jetbot_asset_path, translation=self._jetbot_positions)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Jetbot", get_prim_at_path(jetbot.prim_path), self._sim_config.parse_actor_config("Jetbot"))

    def set_up_scene(self, scene) -> None:

        self.get_jetbot()
        super().set_up_scene(scene)
        self._jetbots = ArticulationView(prim_paths_expr="/World/envs/.*/Jetbot", name="jetbot_view")
        scene.add(self._jetbots)

        return

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self._previous_jetbot_position, jetbot_world_orientation = self._jetbots.get_world_poses()

        actions = actions.to(self._device)

        velocities = torch.zeros((self._jetbots.count, self._jetbots.num_dof), dtype=torch.float32, device=self._device)
        #velocities = self._max_velocity * actions
        print(actions.size(),self._jetbots.num_dof)

        #velocities[:, self._jetbots.num_dof] = self._max_velocity * actions[:0]
        velocities =self._max_velocity * actions
        indices = torch.arange(self._jetbots.count, dtype=torch.int32, device=self._device)

        self._jetbots.set_joint_velocity_targets(velocities, indices=indices)
 
    def get_observations(self) -> dict:

        #self.progress_buf[:] += 1
        #self._my_world.render()
        jetbot_world_position, jetbot_world_orientation = self._jetbots.get_world_poses()
        #shape is (M, 6) linear and angular
        #jetbot_velocity = self._jetbots.get_velocities()
        jetbot_linear_velocity = self._jetbots.get_linear_velocities()
        jetbot_angular_velocity = self._jetbots.get_angular_velocities()
        #print("Lvelocity: "+str(jetbot_linear_velocity))
        #print("Avelocity: "+str(jetbot_angular_velocity))
        #print("velocity: "+str(jetbot_velocity))
        #print("Orienv: "+str(jetbot_velocity[:,2:]))

        #goal_world_position, _ = self.goal.get_world_poses()
        #print("goal_world_position: "+str(goal_world_position))

        self.obs_buf[:, 0] = jetbot_world_position[:, 0]
        self.obs_buf[:, 1] = jetbot_world_position[:, 1]
        self.obs_buf[:, 2] = jetbot_world_position[:, 2]
        self.obs_buf[:, 3] = jetbot_world_orientation[:, 0]
        self.obs_buf[:, 4] = jetbot_world_orientation[:, 1]
        self.obs_buf[:, 5] = jetbot_world_orientation[:, 2]
        self.obs_buf[:, 6] = jetbot_world_orientation[:, 3]
        self.obs_buf[:, 7] = jetbot_linear_velocity[:, 0]
        self.obs_buf[:, 8] = jetbot_linear_velocity[:, 1]
        self.obs_buf[:, 9] = jetbot_linear_velocity[:, 2]
        self.obs_buf[:, 10] = jetbot_angular_velocity[:, 0]
        self.obs_buf[:, 11] = jetbot_angular_velocity[:, 1]
        self.obs_buf[:, 12] = jetbot_angular_velocity[:, 2]
        #self.obs_buf[:, 13] = goal_world_position[:, 0]
        #self.obs_buf[:, 14] = goal_world_position[:, 1]
        #self.obs_buf[:, 15] = goal_world_position[:, 2]

        observations = {
            self._jetbots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def post_reset(self):
        #self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        #self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # randomize all envs
        indices = torch.arange(self._jetbots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
    
    def calculate_metrics(self) -> None:
        current_jetbot_pos_x = self.obs_buf[:, 0]
        current_jetbot_pos_y = self.obs_buf[:, 1]
        previous_jetbot_pos_x = self._previous_jetbot_position[:, 0]
        previous_jetbot_pos_y = self._previous_jetbot_position[:, 1]
        #goal_world_position_x = self.obs_buf[:, 13]
        #goal_world_position_y = self.obs_buf[:, 14]
        goal_world_position_x = self._goal_position[0]
        goal_world_position_y = self._goal_position[1]
        #goal_world_position, _ = self.goal.get_world_poses()
        goal_world_position= 10
        current_jetbot_position, _ = self._jetbots.get_world_poses()
        #goal_world_position = self.obs_buf[:, 9]

        print("current_jetbot_pos : ", current_jetbot_pos_x,",",current_jetbot_pos_y)
        print("\n")
        print("goal_world_position_x: ",goal_world_position_x, ", ", goal_world_position_y)
        print("\n")

        # Calculate previous distance of the jetbots to the goal
        previous_dist_to_goal_x = torch.square(previous_jetbot_pos_x - goal_world_position_x)
        previous_dist_to_goal_y = torch.square(previous_jetbot_pos_y - goal_world_position_y)
        previous_dist_to_goal = torch.sqrt(previous_dist_to_goal_x + previous_dist_to_goal_y)

        # Calculate current distance of the jetbots to the goal
        current_dist_to_goal_x = torch.square(current_jetbot_pos_x - goal_world_position_x)
        current_dist_to_goal_y = torch.square(current_jetbot_pos_y - goal_world_position_y)
        self._current_dist_to_goal = torch.sqrt(current_dist_to_goal_x + current_dist_to_goal_y)

        print("_current_dist_to_goal: ",self._current_dist_to_goal)
        print("\n")

        reward = previous_dist_to_goal - self._current_dist_to_goal
        #reward = -1 * self._current_dist_to_goal

        print("reward: ",reward)
        print("\n")
        #reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        #reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        #reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        cart_pos = self.obs_buf[:, 0]
        pole_pos = self.obs_buf[:, 2]


        resets = torch.where(self._current_dist_to_goal < self._reset_dist, 1, 0)

        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0