from typing import Any, Dict, List, Optional, Type, Union
from gym import Wrapper, spaces
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy
import numpy as np
import pybullet as p
import torch as th
from stable_baselines3.td3.policies import TD3Policy
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor,NatureCNN
from stable_baselines3.common.type_aliases import Schedule

class CnnCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space used in the model. Input images are
    fed through a CNN , the output features are concatenated ("combined").

    :param observation_space:
    :param features_dim: Number of features to output from each CNN submodule. Defaults to
        512 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)

        self.extractor = NatureCNN(self._observation_space['img'], features_dim=features_dim,
                                    normalized_image=normalized_image)

        if 'pos' in self._observation_space.keys():
            self._features_dim +=self._observation_space['pos'].shape[0]
    def forward(self, observations):
        x=self.extractor(observations['img'])

        if 'pos' in self._observation_space.keys():
            x=th.cat([x,observations['pos']],dim=1)
        return x
class RecurrentPPOMultiInputPolicy(RecurrentActorCriticPolicy):
    """
    MultiInputActorClass policy class for the  RecurrentPPO algorithm (has both policy and value prediction).

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch=[256,256],
        activation_fn = nn.ReLU,
        ortho_init = True,
        use_sde = False,
        log_std_init = 0.0,
        full_std = True,
        use_expln = False,
        squash_output = False,
        features_extractor_class = CnnCombinedExtractor,
        features_extractor_kwargs = None,
        share_features_extractor = True,
        normalize_images = True,
        optimizer_class = th.optim.Adam,
        optimizer_kwargs = None,
        lstm_hidden_size=512,
        n_lstm_layers = 1,
        shared_lstm = False,
        enable_critic_lstm = True,
        lstm_kwargs = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )

class TD3MultiInputPolicy(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3 to be used with Dict observation spaces.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = [256,256],
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CnnCombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
class RGBWrapper(Wrapper):
    """
    Wrapper for SurRoL environments, adding and RGB image of the simulation state to the observation space.
    :param env: the original gym environment.
    :param rnd: if true, applies domain randomization to the simulation.
    :param goal: if true, adds the desired goal to the observation space.
    """

    def __init__(self, env,rnd=False, goal=False):
        super(RGBWrapper, self).__init__(env)
        self.rnd=rnd
        self.goal = goal
        obs=self.reset()
        pos = np.zeros(env.action_space.shape[-1])
        self.observation_space = spaces.Dict(observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape,
                                                                    dtype='float32'),
                                             achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape,
                                                                      dtype='float32'),
                                             desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape,
                                                                     dtype='float32'),
                                             img=spaces.Box(low=0, high=255, shape=obs["img"].shape,
                                                            dtype=obs["img"].dtype),
                                             pos=spaces.Box(0.0, 1.0, shape=pos.shape,
                                                            dtype=pos.dtype)
                                             )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        p.changeVisualShape(3, -1, rgbaColor=[0, 0, 0, 0])
        if self.rnd:
            c = np.random.rand(3)
            pos = np.random.rand(3) * 10
            p.configureDebugVisualizer(rgbBackground=c, lightPosition=pos)
            self.c = np.random.rand(6, 4)
            self.c[:, 3] = np.ones(6)
            p.changeVisualShape(5, -1, rgbaColor=self.c[0])
            p.changeVisualShape(4, 4, rgbaColor=self.c[1])
            p.changeVisualShape(4, -1, rgbaColor=self.c[2])
            p.changeVisualShape(1, 15, rgbaColor=self.c[3])
            p.changeVisualShape(2, -1, rgbaColor=self.c[4])
            for i in range(3,8):
                p.changeVisualShape(1, i, rgbaColor=self.c[5])
        obs["img"] = self.env.render('rgb_array')
        obs['pos'] =np.ones(self.env.action_space.shape[-1])*0.5
        if self.goal:
            obs['pos'] = np.concatenate(
                 np.divide(obs["desired_goal"] - self.workspace_limits1[:, 0],
                           self.workspace_limits1[:, 1] - self.workspace_limits1[:, 0]),
                 obs['pos']
                 )
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["img"] = self.env.render('rgb_array')
        obs['pos'] = (np.clip(action, -1, 1)+1)*0.5
        if self.goal:
            obs['pos'] = np.concatenate(
                 np.divide(obs["desired_goal"] - self.workspace_limits1[:, 0],
                           self.workspace_limits1[:, 1] - self.workspace_limits1[:, 0]),
                 obs['pos']
                 )
        done = info['is_success'] or done
        return obs, reward, done, info