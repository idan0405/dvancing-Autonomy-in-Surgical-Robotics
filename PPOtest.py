import gym
from gym.wrappers import TimeLimit
import time
from sb3_contrib import RecurrentPPO
from utils import *

env=gym.make('NeedleReach-v0', render_mode='human')
env=TimeLimit(env, max_episode_steps=50)
env=RGBWrapper(env)
model=RecurrentPPO.load("ppo-reach")

lstm_states = None
episode_starts = np.ones(1, dtype=bool)
ep=1.0
suc=0.0
while True:
    action, lstm_states = model.predict(obs,state=lstm_states,episode_start=episode_starts, deterministic=True)
    obs, rewards, done, info = env.step(action)
    episode_starts=done
    time.sleep(0.025)
    if done:
        if info['is_success']:
            suc+=1.0
        if ep%100==0:
            print(ep)
            print(suc/ep)
        ep+=1
        obs=env.reset()
