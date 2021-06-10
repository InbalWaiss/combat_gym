import gym
import gym_combat
from gym_combat.envs.gym_combat import GymCombatEnv

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def chcek_my_env():
    env = GymCombatEnv()
    check_env(env)

# Instantiate the env


env = make_vec_env('gym-combat-v0', n_envs=4)
model = PPO('MlpPolicy', env, verbose=1,gamma=0.99,learning_rate=0.00001,tensorboard_log=r'.\gym_combat\gym_combat\envs\Arena\statistics\tensorboard',n_steps=32, n_epochs=4, clip_range=0.2, ent_coef=0.025, vf_coef=0.005, clip_range_vf=None)
model.learn(total_timesteps=8000000)
model.save("ppo2_GymCombatEnv_MlpPolicy_10000000")

model = PPO.load("ppo2_GymCombatEnv_MlpPolicy_10000000")
env = GymCombatEnv()
obs = env.reset()
num_of_games = 0
while num_of_games < 100:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
        num_of_games += 1
print("End")