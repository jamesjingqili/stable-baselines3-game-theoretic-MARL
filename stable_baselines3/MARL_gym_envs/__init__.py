"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from typing import Any

from gymnasium.envs.registration import (
    make,
    pprint_registry,
    register,
    registry,
    spec,
)
register(
    id="LQR-v2",
    entry_point="stable_baselines3.MARL_gym_envs.LQR:LQR_Env2",
    max_episode_steps=300,
    reward_threshold=1e9,
)
register(
    id="LQ_game-v3",
    entry_point="stable_baselines3.MARL_gym_envs.LQ_game:LQ_game_Env3",
    max_episode_steps=300,
    reward_threshold=1e9,
)

register(
    id="LQR-v3",
    entry_point="stable_baselines3.MARL_gym_envs.LQR:LQR_Env3",
    max_episode_steps=300,
    reward_threshold=1e9,
)

# # Customized environments begin: 
# register(
#     id="Three_Unicycle_Game-v0",
#     entry_point="MAGPS.MARL_gym_envs.Three_Unicycle:Three_Unicycle_Game_Env0",
#     max_episode_steps=300,
#     reward_threshold=1e9,
# )



# register(
#     id="basketball-v0",
#     entry_point="MAGPS.MARL_gym_envs.Six_basketball_players:basketball_Env0",
#     max_episode_steps=300,
#     reward_threshold=1e9,
# )