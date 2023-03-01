from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole
from omniisaacgymenvs.robots.articulations.jetbot import Jetbot

from omni.isaac.core.utils.nucleus import get_assets_root_path

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math

import omni                                                     # Provides the core omniverse apis
import asyncio                                                  # Used to run sample asynchronously to not block rendering thread

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
        #self.obs_buf = torch.zeros((self._num_envs * self._num_agents , self.num_observations), device=self._device, dtype=torch.float)

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_agents = self._task_cfg["env"]["numAgents"]
        self._jetbot_positions = torch.tensor([0.0, 0.0, 0.1])

        # Get the offset translation
        self._agents_offset_x = self._task_cfg["env"]["agentsOffsetX"]
        self._agents_offset_y = self._task_cfg["env"]["agentsOffsetY"]
        self._agents_offset_z = self._task_cfg["env"]["agentsOffsetZ"]
        self._jetbot_positions_offset = torch.tensor([
            self._agents_offset_x, 
            self._agents_offset_y, 
            self._agents_offset_z
            ])

        # Get the lidar parameters
        self._lidar_min_range = self._task_cfg["lidar"]["min_range"]
        self._lidar_max_range = self._task_cfg["lidar"]["max_range"]
        self._lidar_draw_points = self._task_cfg["lidar"]["draw_points"]
        self._lidar_draw_lines = self._task_cfg["lidar"]["draw_lines"]
        self._lidar_horizontal_fov = self._task_cfg["lidar"]["horizontal_fov"]
        self._lidar_vertical_fov = self._task_cfg["lidar"]["vertical_fov"]
        self._lidar_horizontal_resolution = self._task_cfg["lidar"]["horizontal_resolution"]
        self._lidar_vertical_resolution = self._task_cfg["lidar"]["vertical_resolution"]
        self._lidar_rotation_rate = self._task_cfg["lidar"]["rotation_rate"]
        self._lidar_high_lod = self._task_cfg["lidar"]["high_lod"]
        self._lidar_yaw_offset = self._task_cfg["lidar"]["yaw_offset"]
        self._lidar_enable_semantics = self._task_cfg["lidar"]["enable_semantics"]
        
        self._reset_dist = self._task_cfg["env"]["resetDist"]
        #self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        # Set number of observations per robot
        self._num_observations = 16 + (int)(self._lidar_horizontal_fov / self._lidar_horizontal_resolution)

        # Set number of actions per environment
        self._num_actions = 2 * self._num_agents

        
        self._goal_position = [10.0, 0.0, 3.1]
        self._max_velocity = 10.0      

        RLTask.__init__(self, name, env)
        self.obs_buf = torch.zeros((self._num_envs * self._num_agents , self.num_observations), device=self._cfg["sim_device"], dtype=torch.float)

        return
    
    def get_jetbot(self):

        from pxr import UsdGeom, Gf, UsdPhysics                         # pxr usd imports used to create the cube

        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"

        for i in range(self._num_agents):

            jetbot = Jetbot(
                prim_path=self.default_zero_env_path + "/Jetbot_" + str(i),
                name="Jetbot",
                usd_path=jetbot_asset_path,
                translation=self._jetbot_positions + i * self._jetbot_positions_offset
                )
            self._sim_config.apply_articulation_settings("Jetbot", get_prim_at_path(jetbot.prim_path), self._sim_config.parse_actor_config("Jetbot"))

            #stage = omni.usd.get_context().get_stage()
            #jetbot_prim = stage.GetPrimAtPath(self.default_zero_env_path + "/Jetbot_" + str(i))
            #collisionAPI = UsdPhysics.CollisionAPI.Apply(jetbot_prim) 

    def create_lidars(self):
        from omni.isaac.range_sensor import _range_sensor               # Imports the python bindings to interact with lidar sensor

        stage = omni.usd.get_context().get_stage()                      # Used to access Geometry
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface() # Used to interact with the LIDAR
        #base_prim_path = "/World/envs"
        base_prim_path = "/envs"
        #omni.kit.commands.execute('DeletePhysicsSceneCommand',stage = stage, path='/PhysicsScene')
        #omni.kit.commands.execute('AddPhysicsSceneCommand',stage = stage, path='/World/PhysicsScene')
        
        for i in range(self._num_envs):

            env_path = "/env_" + str(i)

            for j in range(self._num_agents):

                jetbot_path = "/Jetbot_" + str(j)
                parent_prim = base_prim_path + env_path + jetbot_path + "/chassis"
                #lidar_path = jetbot_path + "_lidar"
                lidar_path = "/LidarName"
                result, prim = omni.kit.commands.execute(
                            "RangeSensorCreateLidar",
                            path=lidar_path,
                            parent=parent_prim,
                            min_range=self._lidar_min_range,
                            max_range=self._lidar_max_range,
                            draw_points=self._lidar_draw_points,
                            draw_lines=self._lidar_draw_lines,
                            horizontal_fov=self._lidar_horizontal_fov,
                            vertical_fov=self._lidar_vertical_fov,
                            horizontal_resolution=self._lidar_horizontal_resolution,
                            vertical_resolution=self._lidar_vertical_resolution,
                            rotation_rate=self._lidar_rotation_rate,
                            high_lod=self._lidar_high_lod,
                            yaw_offset=self._lidar_yaw_offset,
                            enable_semantics=self._lidar_enable_semantics
                        )

    def set_up_scene(self, scene) -> None:

        stage = omni.usd.get_context().get_stage()
        #omni.kit.commands.execute('AddPhysicsSceneCommand',stage = stage, path='/World/PhysicsScene')

        self.get_jetbot()
        self.create_lidars()
        super().set_up_scene(scene)
        #self._jetbots = ArticulationView(prim_paths_expr="/World/envs/.*/Jetbot_*", name="jetbot_view")
        self._jetbots = ArticulationView(prim_paths_expr="/envs/.*/Jetbot_*", name="jetbot_view")
        scene.add(self._jetbots)
        #print("AticulationviewPoses:",self._jetbots.get_world_poses())
        #exit()

        return

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self._previous_jetbot_position, jetbot_world_orientation = self._jetbots.get_world_poses()
        new_actions = torch.reshape(actions,(self._jetbots.count, self._jetbots.num_dof))
        #actions = actions.to(self._device)
        new_actions = new_actions.to(self._device)

        #velocities = torch.zeros((self._num_envs, self._jetbots.num_dof * self._num_agents), dtype=torch.float32, device=self._device)
        velocities = torch.zeros((self._jetbots.count, self._jetbots.num_dof), dtype=torch.float32, device=self._device)
        
        #velocities = self._max_velocity * actions
        print("velocities.shape():",velocities.size(), "actions:",new_actions.size(),self._jetbots.num_dof)

        #velocities[:, self._jetbots.num_dof] = self._max_velocity * actions[:0]
        velocities =self._max_velocity * new_actions
        #indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
        indices = torch.arange(self._jetbots.count, dtype=torch.int32, device=self._device)
        print("velocities.shape():",velocities.size(),"indices:",indices.size() )

        self._jetbots.set_joint_velocity_targets(velocities, indices=indices)


    def get_lidar_data(self):
    
        base_prim_path = "/envs"
        distances_buf = torch.empty(self.num_envs * self.num_agents, (int)(self._lidar_horizontal_fov / self._lidar_horizontal_resolution))
    
        for i in range(self._num_envs):

            env_path = "/env_" + str(i)

            for j in range(self._num_agents):

                jetbot_path = "/Jetbot_" + str(j)
                parent_prim = base_prim_path + env_path + jetbot_path + "/chassis"
                #lidar_path = jetbot_path + "_lidar"
                lidar_path = parent_prim + "/LidarName"
                
                distances = self.lidarInterface.get_linear_depth_data(lidar_path)
                distances_torch = torch.from_numpy(distances)
                distances_flat = torch.flatten(distances_torch)
                distances_scaled = (distances_flat - self._lidar_min_range) / (self._lidar_max_range - self._lidar_min_range)
                
                agent_index = i * self.num_agents + j
                distances_buf[agent_index:] = distances_scaled
    
        #base_prim_path = "/World/envs"
        #env_path = "/env_" + str(env_index)
        #jetbot_path = "/Jetbot_" + str(agent_index)
        #parent_prim = base_prim_path + env_path + jetbot_path + "/chassis"
        #lidar_path = parent_prim + "/LidarName"

        #pointcloud = self.lidarInterface.get_point_cloud_data(lidar_path)
        
        print("fgjfjfund", distances_scaled.size())
        

        #print("Point Cloud", pointcloud)
        print("Distances", distances_scaled)
        #print("Point Cloud Shape", distances_scaled.size)
        
        #self.lidar_buf[:] = distances_scaled
        return distances_buf
    
 
    def get_observations(self) -> dict:

        #self.progress_buf[:] += 1
        #self._my_world.render()
        jetbot_world_position, jetbot_world_orientation = self._jetbots.get_world_poses()
        #shape is (M, 6) linear and angular
        #jetbot_velocity = self._jetbots.get_velocities()
        jetbot_linear_velocity = self._jetbots.get_linear_velocities()
        jetbot_angular_velocity = self._jetbots.get_angular_velocities()
        print("Linear vel shape", jetbot_linear_velocity.size())
        print("Linear angular shape", jetbot_angular_velocity.size())
        jetbot_lidar = self.get_lidar_data()
        print("Lidar data", jetbot_lidar)
        print("Lidar shape", jetbot_lidar.size())
        #print("Lvelocity: "+str(jetbot_linear_velocity))
        #print("Avelocity: "+str(jetbot_angular_velocity))
        #print("velocity: "+str(jetbot_velocity))
        #print("Orienv: "+str(jetbot_velocity[:,2:]))
        
        print("Positions x", jetbot_world_position[:, 0])
        print("Positions y", jetbot_world_position[:, 1])

        #goal_world_position, _ = self.goal.get_world_poses()
        #print("goal_world_position: "+str(goal_world_position))
        print("self.obs_buf.size()",self.obs_buf.size(),"jetbot_world_position", jetbot_world_position.size())
        self.obs_buf[:, 0] = jetbot_world_position[:, 0]    # X
        self.obs_buf[:, 1] = jetbot_world_position[:, 1]    # Y
        self.obs_buf[:, 2] = jetbot_world_position[:, 2]    # Z
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
 
        self.obs_buf[:, 16:] = jetbot_lidar
        
        print("Observations", self.obs_buf)

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
        indices = torch.arange(self._jetbots.count/2, dtype=torch.int64, device=self._device)
        print("self._jetbots.count",self._jetbots.count)
        
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

        #self.get_lidar_data(0, 0)

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
