from gym.envs.registration import register

register(
    id='Darts-v0',
    entry_point='gym_Darts.envs:DartsEnv',
)
