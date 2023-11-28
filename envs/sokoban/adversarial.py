import gym
import numpy as np
from . import register
from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast, FieldStates

class SokobanAdversarialEnv(SokobanEnvFast):
    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=1,
                 n_clutter=50,
                 random_z_dim=50, 
                 seed=0):

      ## at least 5x5 map
      super().__init__(dim_room=dim_room, max_steps=120, num_boxes=4, seed = seed)
      self.game_start_room = self.generate_default_room()
      self.restore_full_state_from_np_array_version(self.game_start_room)
      self.adversary_step_count = 0
      self.n_clutter = n_clutter
      #self.seed = seed
      self.random_z_dim = random_z_dim
      self.adversary_max_steps = 50
      self.adversary_action_dim = dim_room[0] * dim_room[1]
      self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
      self.adversary_ts_obs_space = gym.spaces.Box(
        low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
      self.adversary_randomz_obs_space = gym.spaces.Box(
        low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
      self.adversary_image_obs_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(self.dim_room[0], self.dim_room[1], 3),
        dtype='uint8')

      # Adversary observations are dictionaries containing an encoding of the
      # grid, the current time step, and a randomly generated vector used to
      # condition generation (as in a GAN).
      self.adversary_observation_space = gym.spaces.Dict(
          {'image': self.adversary_image_obs_space,
           'time_step': self.adversary_ts_obs_space,
           'random_z': self.adversary_randomz_obs_space})

    @property
    def processed_action_dim(self):
      return 1       

    @property
    def encoding(self):
      return self._internal_state.get_np_array_version()

    def generate_default_room(self):
        """
        Generates basic empty Sokoban room with one box, represented by an integer matrix.
        The elements are encoded in one hot fashion
        :return: Numpy 3d Array
        """
        room = np.zeros(shape = (self.dim_room[0]+2, self.dim_room[1]+2, 7))
        room[:][:][FieldStates.empty] = 1
        for i in range(self.dim_room[0]+2):
            room[i][0][FieldStates.wall] = 1
            room[i][self.dim_room[1]+1][FieldStates.wall] = 1
        for z in range(self.dim_room[1]+2):
            room[0][z][FieldStates.wall] = 1
            room[self.dim_room[0]+1][FieldStates.wall] = 1
        room[2][2][FieldStates.player] = 1
        room[3][3][FieldStates.box] = 1
        room[4][4][FieldStates.box_target] = 1
      
        return room

    def reset_agent(self):
        self.restore_full_state_from_np_array_version(self.game_start_room)
        starting_observation = self.render(render_mode)
        return starting_observation
        
    def reset(self):
        self.game_start_room = self.generate_default_room()
        self.restore_full_state_from_np_array_version(self.game_start_room)
        self.adversary_step_count = 0
        
    def remove_wall(self, x, y):
        if self.game_start_room[x][y][FieldStates.wall] == 1 :
            self.game_start_room[x][y][FieldStates.wall] = 0
            self.game_start_room[x][y][FieldStates.empty] = 1

    def step_adversary(self, loc):
        if loc >= self.adversary_action_dim:
          raise ValueError('Position passed to step_adversary is outside the grid.')

        # Resample block count if necessary, based on first loc
        if self.resample_n_clutter and not self.n_clutter_sampled:
          n_clutter = int((loc/self.adversary_action_dim)*self.n_clutter)
          self.adversary_max_steps = n_clutter + 2
          self.n_clutter_sampled = True

        if self.adversary_step_count < self.adversary_max_steps:
          # Add offset of 1 for outside walls
          x = int(loc % (self.width - 2)) + 1
          y = int(loc / (self.width - 2)) + 1
          done = False

          #if self.choose_goal_last:
          #  should_choose_goal = self.adversary_step_count == self.adversary_max_steps - 2
          #  should_choose_agent = self.adversary_step_count == self.adversary_max_steps - 1
          #else:
          should_choose_goal = self.adversary_step_count == 0
          should_choose_agent = self.adversary_step_count == 2
          should_choose_box = self.adversary_step_count == 1
          # print(f"{self.adversary_step_count}/{self.adversary_max_steps}", flush=True)
          # print(f"goal/agent = {should_choose_goal}/{should_choose_agent}", flush=True)

          # Place goal
          if should_choose_goal:
            # If there is goal noise, sometimes randomly place the goal
            self.game_start_room[x][y] = np.zeros(7) # Remove any walls that might be in this loc
            self.game_start_room[x][y][FieldStates.box_target] = 1

          # Place the agent
          elif should_choose_agent:
            self.game_start_room[x][y] = np.zeros(7)
            self.game_start_room[x][y][FieldStates.player] = 1

          elif should_choose_box:
            self.game_start_room[x][y] = np.zeros(7)
            self.game_start_room[x][y][FieldStates.box] = 1
          
          # Place wall
          elif self.adversary_step_count < self.adversary_max_steps:
            # If there is already an object there, action does nothing
            if self.game_start_room[x][y][FieldStates.empty] == 1:
              self.room[x][y][FieldStates.empty] = 0
              self.room[x][y][FieldStates.wall] = 0
              self.n_clutter_placed += 1
              
        self.adversary_step_count += 1

        # End of episode
        if self.adversary_step_count >= self.n_clutter + 3:
          done = True
          self.reset_metrics()
          self.compute_metrics()
        else:
          done = False

        image = self.render()
        obs = {
            'image': image,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        return obs, 0, done, {}
    
    
    def mutate_level(self, num_edits):
        num_tiles = (self.room_dim[0]-2)*(self.room_dim[1]-2)
        edit_locs = list(set(np.random.randint(0, num_tiles, num_edits)))
        actions = np.random.randint(0, 1, len(edit_locs))
        
        free_mask = self.game_start_room[:][:]
        #free_mask[self.agent_start_pos[1]-1, self.agent_start_pos[0]-1] = False
        #free_mask[self.goal_pos[1]-1, self.goal_pos[0]-1] = False

        for loc, a in zip(edit_locs, actions):
          x = loc % (self.width - 2) + 1
          y = loc // (self.width - 2) + 1

          if(self.game_start_room[x][y][FieldStates.wall] == 1) :
            self.game_start_room[x][y][FieldStates.wall] = 0
            self.game_start_room[x][y][FieldStates.empty] = 1
          elif (self.game_start_room[x][y][FieldStates.empty] == 1):
            self.game_start_room[x][y][FieldStates.wall] = 1
            self.game_start_room[x][y][FieldStates.empty] = 0
  
        # Reset meta info
        
        self.step_count = 0
        self.adversary_step_count = 0
        self.reset_metrics()
        self.compute_metrics()
        self.reset_agent()
        image = self.render()
        obs = {
            'image': image,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        
        return obs


class MediumSokobanAdversarialEnv(SokobanAdversarialEnv):
  def __init__(self, seed=None):
    super().__init__(dim_room=(10, 10),
                 max_steps=240,
                 num_boxes=1,
                 n_clutter=30,
                 seed = seed
                 )

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname


register.register(
    env_id='Sokoban-AdversarialEnv-v0',
    entry_point=module_path + ':MediumSokobanAdversarialEnv',
    max_episode_steps=250,
)
