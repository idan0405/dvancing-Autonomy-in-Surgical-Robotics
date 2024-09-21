import gym
from gym.wrappers import TimeLimit
import time
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
from utils import *

print(torch.cuda.is_available())

if __name__ == "__main__":
    env=gym.make('NeedleReach-v0', render_mode='human')
    env=TimeLimit(env, max_episode_steps=50)
    env=RGBWrapper(env,rnd=True)

    model=RecurrentPPO(
        RecurrentPPOMultiInputPolicy,
        env,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        learning_rate=lambda t:t*2e-4,
        clip_range=lambda t:t*0.2,
        ent_coef=0.01,
        tensorboard_log=f"runs/{time.time()}_reach",
    )

    checkpoint_callback = CheckpointCallback(
      save_freq=10000,
      save_path="./ppo_reach/",
      name_prefix="rl_model",
    )
    model.learn(total_timesteps=5e6,log_interval=1,progress_bar=True, callback=checkpoint_callback)
    model.save("ppo-reach")