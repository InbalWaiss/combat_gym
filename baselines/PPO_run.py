import gym
import gym_combat
from gym_combat.envs.gym_combat import GymCombatEnv
from gym_combat.envs.Common.constants import WinEnum
import os

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def chcek_my_env():
    env = GymCombatEnv()
    check_env(env)

# Instantiate the env
learn = False
if learn:
    env = make_vec_env('gym-combat-v0', n_envs=4)
    tensorboard_path = os.path.join("..", "tensorboard_log")
    model = PPO('MlpPolicy', env, verbose=1,gamma=0.99,learning_rate=0.0001,tensorboard_log=tensorboard_path,n_steps=32, n_epochs=4, clip_range=0.2, ent_coef=0, vf_coef=0.005, clip_range_vf=None)
    model.learn(total_timesteps=10000000, log_interval = 100)
    model.save("ppo2_GymCombatEnv_MlpPolicy_10000000")

model = PPO.load("ppo2_GymCombatEnv_MlpPolicy_10000000")
env = GymCombatEnv()
obs = env.reset()
num_of_games = 0
blue_win_counter = 0
while num_of_games < 2000:

    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #env.render()
    if dones:
        if info['win'] == WinEnum.Blue:
            blue_win_counter += 1
        obs = env.reset()
        num_of_games += 1
        if num_of_games%200 ==0:
            print(blue_win_counter, num_of_games, blue_win_counter/num_of_games)
print ("Blue won in {} games out of {}".format(blue_win_counter, num_of_games))
print("End")