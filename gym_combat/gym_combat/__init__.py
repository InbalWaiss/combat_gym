from gym.envs.registration import register

register(
    id='gym-combat-v0',
    entry_point='gym_combat.gym_combat.envs:GymCombatEnv',
)