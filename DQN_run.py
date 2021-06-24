import gym
import gym_combat
from gym_combat.envs.gym_combat import GymCombatEnv
from gym_combat.envs.Common.constants import WinEnum
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_combat.envs.Common.constants import *

total_timesteps = 20000
checkpoint_freq = 5000

n_games = 1000
tensorboard_path = "tensorboard_log"#os.path.join("..", "tensorboard_log")
checkpoint_path = "checkpoints"#os.path.join("..", "checkpoints")
trained_models_path = "trained_models"#os.path.join("..", "trained_models")
res_path = "res"#os.path.join("..", "res")
checkpoint_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path=checkpoint_path, name_prefix='dqn')

def dqn_train(gamma, lr):
    network_arc = 'MlpPolicy'
    model_name = "dqn_{}_{}M_g_{}_lr_{}".format(network_arc[:3], total_timesteps/1000000, gamma, lr)

    env = GymCombatEnv(run_name = model_name)
    model = DQN(network_arc, env, verbose=1,gamma=gamma,learning_rate=lr,tensorboard_log=tensorboard_path)
    model.learn(total_timesteps=total_timesteps+1000, log_interval = 100, callback=checkpoint_callback, tb_log_name = model_name)
    model.save(os.path.join(trained_models_path, model_name + ".zip"))
    return model_name

def ppo_check_model(model_path, n_games):
    model = DQN.load(model_path)
    env = GymCombatEnv()
    obs = env.reset()
    counter, blue_win_counter = 0,0
    while counter < n_games:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            if info['win'] == WinEnum.Blue:
                blue_win_counter += 1
            obs = env.reset()
            counter += 1
    print ("{}:{} Blue won in {} games out of {}".format(model_path, blue_win_counter/n_games, blue_win_counter, n_games))
    return blue_win_counter/n_games

gamma = 0.99
lr = 0.0003
for lr in [0.0003]:
    res = {}
    trained_model_name = dqn_train(gamma, lr)
    for x in range(checkpoint_freq, total_timesteps+1000, checkpoint_freq):
        model_path = os.path.join(checkpoint_path, "dqn_{}_steps".format(x))
        success_ratio = ppo_check_model(model_path, n_games)
        res[x] = success_ratio
    print(res)
    with open(os.path.join(res_path, trained_model_name+'.txt'), 'w') as f:
        for k in res:
            f.write("{:7d}, {}\n".format(k,res[k]))

