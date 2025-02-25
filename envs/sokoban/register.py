

"""Register Sokoban environments with OpenAI gym."""

import gym
from envs.registration import register as gym_register

env_list = []


def register(env_id, entry_point, reward_threshold=0.95, max_episode_steps=None):
  """Register a new environment with OpenAI gym based on id."""
  assert env_id.startswith("Sokoban-")
  if env_id in env_list:
    del gym.envs.registry.env_specs[id]
  else:
    # Add the environment to the set
    env_list.append(id)

  kwargs = dict(
    id=env_id,
    entry_point=entry_point,
    reward_threshold=reward_threshold
  )

  if max_episode_steps:
    kwargs.update({'max_episode_steps':max_episode_steps})

  # Register the environment with OpenAI gym
  gym_register(**kwargs)
