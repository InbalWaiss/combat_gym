import gym
import gym_combat
from gym_combat.envs.gym_combat import GymCombatEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import DQN
from stable_baselines.common.callbacks import CheckpointCallback
from gym_combat.envs.Common.constants import *
import tensorflow as tf
import torch as th
import torch.nn as nn

def chcek_my_env():
    env = GymCombatEnv()
    check_env(env)


# Instantiate the env
env = GymCombatEnv()
# Define and Train the agent

checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./logs/',
                                         name_prefix='rl_model')

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=6, stride=3, padding=0),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.Flatten(),
        )
        # h4 = Dense(512, activation='elu', name="fc")(context)
        # output = Dense(num_actions, name="output")(h4)


        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
)

# model = DQN('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=3000000, callback=checkpoint_callback)
# model.save("deepq_GymCombatEnv_MlpPolicy_15X15_3000000")
# model = DQN.load("deepq_GymCombatEnv_MlpPolicy_15X15_3000000")

env = GymCombatEnv()
#policy_kwargs = dict(act_fun=tf.nn.elu, net_arch=[256, 128, 128])
#model = DQN('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
model = DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("deepq_GymCombatEnv_CnnPolicy_1000000_Berlin")
model = DQN.load("deepq_GymCombatEnv_CnnPolicy_1000000_Berlin")


# env = GymCombatEnv()
# model = DQN('LnMlpPolicy', env, verbose=1)
# model.learn(total_timesteps=1000000)
# model.save("deepq_GymCombatEnv_LnMlpPolicy_1000000_15X15")
#model = DQN.load("deepq_GymCombatEnv_LnMlpPolicy_1000000_15X15")

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