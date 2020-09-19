from gym.envs.registration import register

register(
    id='sat3-v0',
    entry_point='gym_sat3.envs:Sat3Env',
)
register(
    id='sat3-extrahard-v0',
    entry_point='gym_sat3.envs:Sat3ExtraHardEnv',
)