
import torch
import torch.nn as nn

import numpy as np

from torch.distributions.categorical import Categorical
from dataclasses import dataclass

import math


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

        agents_x = x[:, :self.agents, :]
        object_x = x[:, self.agents:, :]

        agents_x = x.reshape(x.shape[0], -1)

        logits = self.actor(agents_x)

        logits = logits.reshape(x.shape[0], self.agents, -1)

        # logits = self.params.expand(x.shape[0], self.agents, 7)

        logits[~mask] = -float('inf')

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.critic(agents_x)

        return action, probs.log_prob(action), probs.entropy(), value


@dataclass
class AgentConfig:
    n_embed: int = 64
    dropout: float = 0.0
    n_head: int = 4
    n_layer: int = 1
    bias: bool = True


class MLP(nn.Module):

    def __init__(self, config, inp_size=None):
        super().__init__()
        if inp_size is None:
            inp_size = config.n_embed
        self.c_fc = nn.Linear(
            inp_size, 4 * config.n_embed, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        assert self.config.n_embed % self.config.n_head == 0, f"Hidden size {config.n_embed} must be divisible by number of attention heads {heads}"

        self.q = nn.Linear(self.config.n_embed, self.config.n_embed)
        self.k = nn.Linear(self.config.n_embed, self.config.n_embed)
        self.v = nn.Linear(self.config.n_embed, self.config.n_embed)
        self.out_lin = nn.Linear(self.config.n_embed, self.config.n_embed)
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.sa_layer_norm = nn.LayerNorm([self.config.n_embed])
        self.out_layer_norm = nn.LayerNorm([self.config.n_embed])

        self.dim_per_head = (self.config.n_embed // self.config.n_head)

        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor):

        # print(x.shape)
        bs, seq_len, _ = x.shape

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, x.shape[1], self.config.n_head, -1).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, seq_len, self.config.n_head * self.dim_per_head)

        q_vals = shape(self.q(x))
        k_vals = shape(self.k(x))
        v_vals = shape(self.v(x))

        tmp = self.q(x)
        tmp2 = unshape(shape(tmp))
        assert (tmp-tmp2).abs().max() < 1e-5

        # (bs, n_heads, seq_len, dim_per_head)
        q_vals = q_vals / math.sqrt(self.dim_per_head)
        # (bs, n_heads, seq_len, seq_len)
        scores = torch.matmul(q_vals, k_vals.transpose(2, 3))
        # (bs, n_heads, seq_length, seq_length)
        weights = torch.softmax(scores, dim=-1)
        # (bs, n_heads, seq_length, seq_length)
        weights = self.dropout(weights)
        # (bs, n_heads, seq_length, dim_per_head)
        res = torch.matmul(weights, v_vals)
        res = unshape(res)  # (bs, seq_length, hidden)

        return res


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Model(nn.Module):
    def __init__(self, envs, config, agents=1, n_action=7, n_features=4):
        super().__init__()

        self.agents = agents
        self.config = config
        self.n_action = n_action

        self.agent_prep = nn.Sequential(
            layer_init(nn.Linear(n_features, config.n_embed)),
            nn.GELU(),
        )
        self.object_prep = nn.Sequential(
            layer_init(nn.Linear(n_features, config.n_embed)),
            nn.GELU(),
        )

        self.atts = nn.ModuleList([Block(config)
                                   for _ in range(config.n_layer)])

        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(self.config.n_embed, self.config.n_embed)),
            nn.GELU(),
            layer_init(nn.Linear(self.config.n_embed, self.config.n_embed)),
            nn.GELU(),
            layer_init(nn.Linear(self.config.n_embed, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(self.config.n_embed, self.config.n_embed)),
            nn.GELU(),
            layer_init(nn.Linear(self.config.n_embed, self.config.n_embed)),
            nn.GELU(),
            layer_init(
                nn.Linear(self.config.n_embed, self.n_action), std=0.01),
        )

        self.post_concat_mlp = MLP(config, inp_size=config.n_embed * 2)

        # Encodings for agents and objects to distinguish them in attention
        self.agent_encoding = nn.Parameter(torch.randn(1, 1, config.n_embed))
        self.object_encoding = nn.Parameter(torch.randn(1, 1, config.n_embed))

    def get_embedding(self, obs):

        A = obs[:, :self.agents, :]
        O = obs[:, self.agents:, :]

        agent_x = self.agent_prep(A)
        object_x = self.object_prep(O)

        # find objects each agent is holding
        idx = obs[:, :self.agents, 3].long().unsqueeze(-1)  # (bs, agents, 1)
        mask = (idx != -1).float()  # (bs, agents, 1)
        idx[idx == -1] = 0

        # pick out the object embeddings
        picked_objects = torch.gather(
            object_x, 1, idx.repeat(1, 1, self.config.n_embed))

        # mask out the objects for empty agents
        picked_objects = picked_objects * \
            mask.repeat(1, 1, self.config.n_embed)

        # concat agent and object embeddings
        # (bs, agents, 2*embed)
        x = torch.cat([agent_x, picked_objects], dim=-1)

        x = self.post_concat_mlp(x)

        x = self.agent_encoding.repeat(x.shape[0], 1, 1) + x

        # TODO: do the same for objects
        y = self.object_encoding.repeat(x.shape[0], 1, 1) + object_x

        x = torch.cat([x, y], dim=1)

        for att in self.atts:
            x = att(x)

        return x

    def get_action_and_value(self, obs, mask, action=None):

        x = self.get_embedding(obs)
        agent_x = x[:, :self.agents, :]

        logits = self.actor(agent_x)

        logits = logits.reshape(x.shape[0], self.agents, -1)

        logits[~mask] = -float('inf')

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.critic(agent_x).squeeze(-1)

        return action, probs.log_prob(action), probs.entropy(), value
