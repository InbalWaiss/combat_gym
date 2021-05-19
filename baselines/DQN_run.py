import gym
import gym_combat
from gym_combat.envs.gym_combat import GymCombatEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import DQN
import tensorflow as tf

def chcek_my_env():
    env = GymCombatEnv()
    check_env(env)


# Instantiate the env
# env = GymCombatEnv()
# Define and Train the agent
#model = DQN('MlpPolicy', env).learn(total_timesteps=5000)



# model = DQN('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=300000)
# model.save("deepq_GymCombatEnv_MlpPolicy")


env = GymCombatEnv()
#policy_kwargs = dict(act_fun=tf.nn.elu, net_arch=[256, 128, 128])
model = DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("deepq_GymCombatEnv_CnnPolicy_1000000_Berlin")
model = DQN.load("deepq_GymCombatEnv_CnnPolicy_1000000_Berlin")


# env = GymCombatEnv()
# model = DQN('LnMlpPolicy', env, verbose=1)
# model.learn(total_timesteps=1000000)
# model.save("deepq_GymCombatEnv_LnMlpPolicy_1000000_Berlin")
# model = DQN.load("deepq_GymCombatEnv_LnMlpPolicy_1000000_Berlin")

env = GymCombatEnv()
obs = env.reset()
num_of_games = 0
while num_of_games<100:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
        num_of_games+=1

print("End")