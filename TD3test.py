import gym
from gym.wrappers import TimeLimit
import time
from stable_baselines3 import TD3
from utils import *
env=gym.make('NeedleReach-v0', render_mode='human')
env=TimeLimit(env, max_episode_steps=50)
env=RGBWrapper(env)
obs = env.reset()
model=TD3.load("td3-reach",env=env)

ep=1.0
suc=0.0
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    time.sleep(0.025)
    if done:
        if info['is_success']:
            suc+=1.0
        j=0
        if ep%100==0:
            print("episode:",ep)
            print("success rate:",suc/ep)
        ep+=1
        obs=env.reset()
