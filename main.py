# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import pygame
import argparse
import os
import random
import time
# from distutils.util import strtobool


import gymnasium as gym
import numpy as np
import torch

from datetime import datetime

import warehouse_env
from agent import Agent, Model, AgentConfig
from ppo import PPO
from utils import make_env, strtobool

# set torch to print all tensor
torch.set_printoptions(profile="full")


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.95,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advenv_antage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.1,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--save-interval", type=int, default=512,
        help="the interval to save the model")
    parser.add_argument("--load", type=str, default=None,
                            help="the model to load if any")
    parser.add_argument("--eval", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, evaluation runs will be deterministic and performed on the CPU")
    parser.add_argument("--show", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, the environment will be rendered")

    # ===============================
    # Custom arguments
    # ===============================
    parser.add_argument("--agents", type=int, default=1,
        help="the number of agents in the environment")

    parser.add_argument("--objects", type=int, default=1,
        help="the number of objects in the environment")

    parser.add_argument("--grid-size", type=int, default=10,
        help="the size of the grid in the environment")

    # ===============================
    # ===============================


    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


if __name__ == "__main__":

    args = parse_args()
    readable_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"{args.env_id}__N:{args.grid_size}__agents:{args.agents}__objects:{args.objects}__{args.seed}__{readable_time}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # define env kwargs
    kwargs = {
        "n_agents": args.agents,
        "n_objects": args.objects,
        "grid_size": args.grid_size,
    }

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, **kwargs)
         for i in range(args.num_envs)]
    )

    test_envs = make_env(args.env_id, args.seed + args.num_envs + 1,
                         args.num_envs + 1, args.capture_video, run_name, **kwargs)()

    assert isinstance(envs.single_action_space,
                      gym.spaces.MultiDiscrete), "only multi discrete action space is supported (one discrete action per agent)"

    # agent = Agent(envs, agents=args.agents)
    agent = Model(envs, config=AgentConfig, agents=args.agents, n_action=7)

    ppo = PPO(agent, envs, test_envs, args, run_name)

    if args.load is not None:
        ppo.load(args.load)
        print("Loaded model from: ", args.load)

    if args.eval:
        ppo.eval()
    elif args.show:
        ppo.play_trajectory(2)
    else:
        ppo.train()
