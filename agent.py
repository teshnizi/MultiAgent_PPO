
import torch
import torch.nn as nn

import numpy as np

from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, agents=1):
        super().__init__()

        self.agents = agents
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape[-1] * 3), 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, self.agents), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape[-1] * 3), 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 64)),
            nn.GELU(),
            layer_init(
                nn.Linear(64, 7 * self.agents), std=0.01),
        )

    def get_action_and_value(self, x, mask, action=None):
        """
        Returns:
            action: (batch_size, xxx, agents)
            log_prob: (batch_size, xxx, agents)
            entropy: (batch_size, xxx, agents)
            value: (batch_size, xxx, agents)
        """

        # agents_x = x[:, :self.agents, :]

        agents_x = x.reshape(x.shape[0], -1)

        if x.shape[0] < 2:
            print(agents_x)
        # print(x.shape, agents_x.shape)
        # 1/0

        logits = self.actor(agents_x)

        logits = logits.reshape(x.shape[0], self.agents, -1)

        # logits = self.params.expand(x.shape[0], self.agents, 7)

        logits[~mask] = -float('inf')

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.critic(agents_x)

        return action, probs.log_prob(action), probs.entropy(), value
