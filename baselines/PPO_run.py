import gym
import gym_combat
from gym_combat.envs.gym_combat import GymCombatEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, DQN

def chcek_my_env():
    env = GymCombatEnv()
    check_env(env)


# Instantiate the env
env = GymCombatEnv()

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000000)
model.save("ppo2_GymCombatEnv_MlpPolicy_10000000")

# env = GymCombatEnv()
# model = PPO2('CnnPolicy', env, verbose=1)
# model.learn(total_timesteps=1000000)
# model.save("ppo2_GymCombatEnv_CnnPolicy_1000000")

# env = GymCombatEnv()
# model = PPO2('CnnLstmPolicy', env, verbose=1)
# model.learn(total_timesteps=100000)
# model.save("ppo2_GymCombatEnv_CnnLstmPolicy")

model = PPO2.load("ppo2_GymCombatEnv_MlpPolicy_10000000")


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