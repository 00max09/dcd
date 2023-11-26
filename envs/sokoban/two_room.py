class TwoRoomsSokobanEnv(SokobanEnvFast):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(max_steps=200)
        self.game_start_room = self._generate_grid()

        direction_obs_space = gym.spaces.Box(
            low=0, high=3, shape=(1,), dtype='uint8')

        self.observation_space = spaces.Dict({
            'image': self.observation_space['image'],
            'direction': direction_obs_space
        })

    def _gen_grid(self, ):
        # Create the grid
        room = np.zeros(shape = (10+2, 10+2, 7))
        room = np.ones(shape=self.room_dim)
        for i in range(self.room_dim[0]+2):
            room[i][0][FieldStates.wall] = 1
            room[i][self.room_dim[1]+1][FieldStates.wall] = 1
        for z in range(self.room_dim[1]+2):
            room[0][z][FieldStates.wall] = 1
            room[self.room_dim[0]+1][FieldStates.wall] = 1
        room[2][2][FieldStates.player] = 1
        room[3][3][FieldStates.box] = 1
        room[9][9][FieldStates.box_target] = 1
        for z in range(1,5) :
            room[5][z][FieldStates.wall] = 1
        for z in range(6,10) :
            room[5][z][FieldStates.wall] = 1
        return room

     def reset(self):
        self.game_start_room = self._gen_grid()
        self.restore_full_state_from_np_array_version(self.game_start_room)
        self.adversary_step_count = 0
        
if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    id='Sokoban-TwoRooms-v0',
    entry_point=module_path+':TwoRoomsSokobanEnv'
)
