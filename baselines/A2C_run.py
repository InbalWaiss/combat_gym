import gym
import gym_combat
from gym_combat.envs.gym_combat import GymCombatEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common import make_vec_env
from stable_baselines3 import A2C

def chcek_my_env():
    env = GymCombatEnv()
    check_env(env)


# Instantiate the env
env = GymCombatEnv()

# model = A2C(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=100000)
# model.save("A2C_GymCombatEnv_MlpPolicy")

env = GymCombatEnv()
model = A2C('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("A2C_GymCombatEnv_CnnPolicy_1000000")

# env = GymCombatEnv()
# model = A2C('CnnLstmPolicy', env, verbose=1)
# model.learn(total_timesteps=100000)
# model.save("A2C_GymCombatEnv_CnnLstmPolicy")

model = A2C.load("a2c_GymCombatEnv")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# # Define and Train the agent
# model = A2C('MlpPolicy', env).learn(total_timesteps=5000)
#
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
