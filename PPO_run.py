import gym
import gym_combat.gym_combat
from gym_combat.gym_combat.envs.Common.constants import WinEnum
import os
import time
import imageio

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
total_timesteps = 20000000
checkpoint_freq = 1000000

n_games = 1000
tensorboard_path = "tensorboard_log"
checkpoint_path = "checkpoints"
trained_models_path = "trained_models"
res_path = "res"
video_path = "videos"


class EnvNum():
    def __init__(self):
        self.n=0
    def get(self):
        self.n += 1
        return self.n


def ppo_train(gamma, lr, vf_coef, ent_coef, train = True):

    network_arc = 'CnnPolicy'
    model_name = "ppo_{}_{}M_g_{}_lr_{}_vfc_{}_entc_{}".format(network_arc[:3], total_timesteps/1000000, gamma, lr, vf_coef, ent_coef)
    if not train:
        return model_name
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_freq/n_envs, save_path=checkpoint_path, name_prefix='ppo')
    env_num = EnvNum()
    print ("trainng model", model_name)
    env = make_vec_env('gym-combat-v0', n_envs=n_envs, env_kwargs={"run_name": model_name, "env_num": env_num}, seed = 0)
    
    model_path = os.path.join(trained_models_path, model_name)
    #model = PPO.load(model_path, env = env)
    model = PPO(network_arc, env, verbose=1,gamma=gamma,learning_rate=lr,tensorboard_log=tensorboard_path, n_steps=32, n_epochs=4, clip_range=0.25, ent_coef=ent_coef, vf_coef=vf_coef, clip_range_vf=None)
    model.learn(total_timesteps=total_timesteps+1000, log_interval = 100, callback=checkpoint_callback, tb_log_name = model_name)
    #model_name = "ppo_{}_{}M_g_{}_lr_{}_vfc_{}_entc_{}".format(network_arc[:3], 2*total_timesteps/1000000, gamma, lr, vf_coef, ent_coef)
    model.save(os.path.join(trained_models_path, model_name + ".zip"))
    return model_name

def ppo_check_model(model_path, model_name, n_games, save_video = False):
    model = PPO.load(os.path.join(model_path, model_name))
    env = gym.make('gym-combat-v0', train_mode = False )
    obs = env.reset()
    counter, blue_win_counter = 0,0
    steps = 0
    if save_video:
        images = []
        img = env.render()
        for _ in range(10):
            images.append(img)
    while counter < n_games:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        steps+=1
        if save_video:
            images.append(env.render())
        if dones:
            if info['win'] == WinEnum.Blue:
                blue_win_counter += 1
            if save_video:
                img = env.render()
                for _ in range(30):
                    images.append(img)
                filename = model_name +"_game"+ str(counter) + ".mp4"
                imageio.mimsave(os.path.join(video_path, filename), images, fps = 3)
            obs = env.reset()
            counter += 1
            if counter%500 ==0:
                print(blue_win_counter, counter, steps/counter)
            if save_video:
                images = []
                img = env.render()
                for _ in range(10):
                    images.append(img) 
    print ("{}:{} Blue won in {} games out of {}".format(model_path, blue_win_counter/n_games, blue_win_counter, n_games))
    return blue_win_counter/n_games, steps/n_games


gamma = 0.95
lr = 0.0001
vf_coef = 0.15
ent_coef = 0
for vf_coef in [0.15,0.25]:
    res = {}
    t0 = time.time()
    trained_model_name = ppo_train(gamma, lr, vf_coef, ent_coef)
    t1 = time.time()
    print("starting tests:")
    for x in range(checkpoint_freq, total_timesteps+1000, checkpoint_freq):
        res[x] = ppo_check_model(checkpoint_path, "ppo_{}_steps".format(x), n_games)
    res[trained_model_name] = ppo_check_model(trained_models_path, trained_model_name, 10000)
    t2 = time.time()
    print(res)
    with open(os.path.join(res_path, trained_model_name+'.txt'), 'w') as f:
        for k in res:
            f.write("{}, {}\n".format(k,res[k]))
    
    print("train time: %f minutes" % ((t1 - t0) / 60))
    print("test time: %f minutes" % ((t2 - t1) / 60))
    #saving_videos
    ppo_check_model(trained_models_path, trained_model_name,n_games = 100, save_video = True)
    


