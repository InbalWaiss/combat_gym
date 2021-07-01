import gym
import gym_combat.gym_combat
from gym_combat.gym_combat.envs.Common.constants import WinEnum
import os
import time

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO, DQN

# #Inbal sanity check
# env = GymCombatEnv()
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=10000000)
# model.save("ppo_GymCombatEnv_MlpPolicy_10000000")



n_envs = 4
total_timesteps = 20000
checkpoint_freq = 5000

n_games = 1000
tensorboard_path = "tensorboard_log"#os.path.join("..", "tensorboard_log")
checkpoint_path = "checkpoints"#os.path.join("..", "checkpoints")
trained_models_path = "trained_models"#os.path.join("..", "trained_models")
res_path = "res"#os.path.join("..", "res")
checkpoint_callback = CheckpointCallback(save_freq=checkpoint_freq/n_envs, save_path=checkpoint_path, name_prefix='ppo')

class EnvNum():
    def __init__(self):
        self.n=0
    def get(self):
        self.n += 1
        return self.n


def ppo_train(gamma, lr, vf_coef):

    network_arc = 'MlpPolicy'
    model_name = "ppo_{}_{}M_g_{}_lr_{}_vfc_{}".format(network_arc[:3], total_timesteps/1000000, gamma, lr, vf_coef)
    env_num = EnvNum()
    env = make_vec_env('gym-combat-v0', n_envs=n_envs, env_kwargs={"run_name": model_name, "env_num": env_num}, seed = 0)

    model = PPO(network_arc, env, verbose=1,gamma=gamma,learning_rate=lr,tensorboard_log=tensorboard_path,n_steps=32, n_epochs=4, clip_range=0.2, ent_coef=0, vf_coef=vf_coef, clip_range_vf=None)
    model.learn(total_timesteps=total_timesteps+1000, log_interval = 100, callback=checkpoint_callback, tb_log_name = model_name)
    model.save(os.path.join(trained_models_path, model_name + ".zip"))
    return model_name

def ppo_check_model(model_path, n_games):
    model = PPO.load(model_path)
    env = gym.make('gym-combat-v0', train_mode = False )
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
vf_coef = 0.1
for vf_coef in [0.1]:
    res = {}
    t0 = time.time()
    trained_model_name = ppo_train(gamma, lr, vf_coef)
    t1 = time.time()
    print("starting tests:")
    for x in range(checkpoint_freq, total_timesteps+1000, checkpoint_freq):
        model_path = os.path.join(checkpoint_path, "ppo_{}_steps".format(x))
        success_ratio = ppo_check_model(model_path, n_games)
        res[x] = success_ratio
    print(res)
    with open(os.path.join(res_path, trained_model_name+'.txt'), 'w') as f:
        for k in res:
            f.write("{:7d}, {}\n".format(k,res[k]))
    t2 = time.time()
    print("train time: %f minutes" % ((t1 - t0) / 60))
    print("test time: %f minutes" % ((t2 - t1) / 60))


