import gym
import gym_combat.gym_combat
from gym_combat.gym_combat.envs.Common.constants import WinEnum
import os
import time
import imageio

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO


def check_environment_creation():
    # Inbal creating env sanity: check that the environment was installed correctly
    # if not working- try "pip install -e gym-combat-v0" in terminal
    from gym_combat.gym_combat.envs.gym_combat import GymCombatEnv
    env = GymCombatEnv()
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000000)
    model.save("ppo_GymCombatEnv_MlpPolicy_10000000")

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

#PPO model params
n_envs =4
batch_size = 2048
n_steps=128
n_epochs=4
clip_range=0.25
n_games = 1000

prev_timesteps =  0 #100000000 # if we want to keep training an old model
total_timesteps = 200000000
checkpoint_freq = 5000000

# folders
tensorboard_path = "tensorboard_log"
create_folder(tensorboard_path)
checkpoint_path = "checkpoints"
create_folder(checkpoint_path)
trained_models_path = "trained_models"
create_folder(trained_models_path)
res_path = "res"
create_folder(res_path)
video_path = "videos"
create_folder(video_path)


class EnvNum():
    def __init__(self):
        self.n=0
    def get(self):
        self.n += 1
        return self.n


def ppo_train(gamma, lr, vf_coef, ent_coef, train = True, mp = 0.01):

    network_arc = 'CnnPolicy' #conv1: 32, conv2: 64, conv3: 64, fc: 512
    model_name = "smart_vs_ppo_{}_{}M_g_{}_lr_{}_vfc_{}_entc_{}_mp_{}_lost0.5".format(network_arc[:3], total_timesteps/1000000, gamma, lr, vf_coef, ent_coef, mp)
    if prev_timesteps > 0:
        model_name = "smart_vs_ppo_{}_{}M_g_{}_lr_{}_vfc_{}_entc_{}_mp_{}_lost0.5_-1".format(network_arc[:3], prev_timesteps/1000000, gamma, lr, vf_coef, ent_coef, mp)

    if not train:
        return model_name
    checkpoint_callback = CheckpointCallback(save_freq=int(checkpoint_freq/n_envs), save_path=checkpoint_path, name_prefix='ppo')

    # creating environment
    env_num = EnvNum()
    env = make_vec_env('gym-combat-v0', n_envs=n_envs, env_kwargs={"run_name": model_name, "env_num": env_num, "move_penalty": mp}, seed = 0)

    # loading model
    model_path = os.path.join(trained_models_path, model_name)
    if prev_timesteps > 0:
        print ("loading model", model_name)
        model = PPO.load(model_path, env = env)
        #model_name = "ppo_greedy_25000000_steps"
        #model_name = "greedy_vs_ppo_{}_{}M_g_{}_lr_{}_vfc_{}_entc_{}_mp_{}_lost0.5_-1".format(network_arc[:3], (prev_timesteps + total_timesteps)/1000000, gamma, lr, vf_coef, ent_coef, mp)
        model_name = "smart_vs_ppo_{}_{}M_g_{}_lr_{}_vfc_{}_entc_{}_mp_{}_lost0.5_-1".format(network_arc[:3], (prev_timesteps + total_timesteps)/1000000, gamma, lr, vf_coef, ent_coef, mp)
    else:
        model = PPO(network_arc, env, verbose=1,gamma=gamma,learning_rate=lr,tensorboard_log=tensorboard_path, n_steps=n_steps, batch_size = batch_size, n_epochs=n_epochs, clip_range=clip_range, ent_coef=ent_coef, vf_coef=vf_coef, clip_range_vf=None)

    # training model
    print("training model", model_name)
    model.learn(total_timesteps=total_timesteps+1000, log_interval = 20, callback=checkpoint_callback, tb_log_name = model_name)

    # saving trained model
    model.save(os.path.join(trained_models_path, model_name + ".zip"))
    return model_name


def ppo_check_model(model_path, model_name, n_games, save_video = False, mp = -0.1):
    # check model model_name and save videos
    model = PPO.load(os.path.join(model_path, model_name))
    env = gym.make('gym-combat-v0', train_mode = False, move_penalty = mp ) # creating gym environment
    obs = env.reset()
    counter, blue_win_counter, red_win_counter, nowin_win_counter = 0,0,0,0
    steps = 0
    if save_video:
        images = []
        img = env.render()
        for _ in range(10):
            images.append(img)

    # Play the game!
    while counter < n_games:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        steps+=1

        if save_video:
            images.append(env.render())

        if dones: # if terminal state update win counters
            winner = "_no_win"
            if info['win'] == WinEnum.Blue:
                blue_win_counter += 1
                winner = "_blue"
            elif info['win'] == WinEnum.Red:
                red_win_counter += 1
                winner = "_red"
            elif info['win'] == WinEnum.Tie:
                nowin_win_counter += 1

            if save_video:
                img = env.render()
                for _ in range(30):
                    images.append(img)
                filename = model_name +"_game"+ str(counter) + winner + ".mp4"
                imageio.mimsave(os.path.join(video_path, filename), images, fps = 3)

            obs = env.reset() # reset game
            counter += 1
            if counter%500 ==0:
                print(blue_win_counter, red_win_counter, counter, steps/counter)

            if save_video:
                images = []
                img = env.render()
                for _ in range(10):
                    images.append(img)

    print ("{}: success rate:{} Blue:{} Red:{} No win:{} out of {} games".format(model_name, blue_win_counter/n_games, blue_win_counter, red_win_counter, nowin_win_counter, n_games))
    return blue_win_counter/n_games, red_win_counter/n_games, nowin_win_counter/n_games, steps/n_games


gamma = 0.985
lr = 0.0002
vf_coef = 0.1
ent_coef = 0

for mp in [-0.01]: # mp = move penalty
    res = {}
    t0 = time.time()
    trained_model_name = ppo_train(gamma, lr, vf_coef, ent_coef, mp = mp, train = True)

    #trained_model_name = "ppo_250000000_baka_thicken_steps_g99_same_statr_points"
    t1 = time.time()
    print("starting tests:")
    for x in range(checkpoint_freq, total_timesteps+1000, checkpoint_freq):
        res[x] = ppo_check_model(checkpoint_path, "ppo_{}_steps".format(x), n_games, mp=mp)
    res[trained_model_name] = ppo_check_model(trained_models_path, trained_model_name, 10*n_games, mp = mp)
    t2 = time.time()
    print(res)
    with open(os.path.join(res_path, trained_model_name+'.txt'), 'w') as f:
        for k in res:
            f.write("{}, {}\n".format(k,res[k]))
    
    print("train time: %f minutes" % ((t1 - t0) / 60))
    print("test time: %f minutes" % ((t2 - t1) / 60))

    #check model and save videos
    ppo_check_model(trained_models_path, trained_model_name,n_games = n_games, save_video = True, mp = mp)
    


