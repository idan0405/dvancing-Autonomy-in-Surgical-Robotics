import gym
from gym.wrappers import TimeLimit
import time
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from utils import *

if __name__ == "__main__":
    env=gym.make('NeedleReach-v0', render_mode='human')
    env=TimeLimit(env, max_episode_steps=50)
    env=RGBWrapper(env, rnd=True)


    n_actions=env.action_space.shape[-1]
    action_noise=NormalActionNoise(mean=np.zeros(n_actions),sigma=0.1*np.ones(n_actions))
    model = TD3(
       TD3MultiInputPolicy,
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
        ),
        learning_starts=10000,
        learning_rate=lambda t:3e-5*t,
        action_noise=action_noise,
        verbose=0,
        tensorboard_log=f"runs/{time.time()}_reach",
        batch_size=256,
        gradient_steps=50,
        train_freq=(100,"step"),
        buffer_size=200000,
        device="auto",
    )

    i=0
    ep=0
    while i<25000:
        done=False
        obs=env.reset()
        obs["img"] = np.expand_dims(obs["img"], axis=0).transpose((0,3,1,2))
        while not done:
            action=env.get_oracle_action(obs)
            last_obs=obs
            obs, rewards, done, info = env.step(action)
            obs["img"]=np.expand_dims(obs["img"], axis=0).transpose((0,3,1,2))
            model.replay_buffer.add(last_obs,obs,action,rewards,[done],[info])
            i+=1
        ep+=1
        if ep%100==0:
            print(ep)
            print(i)
            print(rewards)

    checkpoint_callback = CheckpointCallback(
      save_freq=10000,
      save_path="./td3_reach/",
      name_prefix="rl_model",
    )

    model.learn(total_timesteps=6e5,log_interval=1,progress_bar=True, callback=checkpoint_callback)
    model.save("td3-reach")