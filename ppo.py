import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.optim as optim
import numpy as np

import time
import os

import pygame


class PPO():

    def __init__(self, agent, envs, test_envs, args, run_name):
        super().__init__()

        self.args = args

        self.device = torch.device(
            "cuda:1" if torch.cuda.is_available() and self.args.cuda else "cpu")

        print("running on device:", self.device)
        self.agent = agent.to(self.device)
        self.envs = envs
        self.test_envs = test_envs
        self.run_name = run_name

        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        if self.args.tensorboard:
            self.writer = SummaryWriter(f"runs/{run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % (
                    "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )

        self.args.num_updates = self.args.total_timesteps // self.args.batch_size

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) +
                               self.envs.single_observation_space.shape).to(self.device)
        self.masks = torch.ones(
            (self.args.num_steps, self.args.num_envs, self.args.agents, 7), dtype=torch.bool).to(self.device)

        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) +
                                   envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros(
            (self.args.num_steps, self.args.num_envs,  self.args.agents)).to(self.device)
        self.rewards = torch.zeros(
            (self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros(
            (self.args.num_steps, self.args.num_envs)).to(self.device)
        self.values = torch.zeros(
            (self.args.num_steps, self.args.num_envs,  self.args.agents)).to(self.device)
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        self.returns = torch.zeros_like(self.rewards).to(self.device)

        self.next_done = -1
        self.next_obs = -1
        self.next_mask = -1

    def collect(self, steps):

        # copy the latest envs state
        next_obs = self.next_obs
        next_done = self.next_done
        next_mask = self.next_mask

        for step in range(0, steps):

            self.global_step += 1 * self.args.num_envs
            self.obs[step] = next_obs
            self.masks[step] = next_mask
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(
                    next_obs, next_mask, action=None)

                self.values[step] = value

            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _, info = self.envs.step(
                action.cpu().numpy())

            self.rewards[step] = torch.Tensor(reward).to(self.device)

            next_mask = np.stack(info['mask'], axis=0)
            next_mask = torch.tensor(next_mask).to(self.device)
            next_obs = torch.Tensor(next_obs).to(self.device)
            next_done = torch.Tensor(done).to(self.device)

            if 'final_info' in info.keys():
                for item in info['final_info']:
                    if item != None and "episode" in item.keys():
                        print(
                            f"global_step={self.global_step}, episodic_return={item['episode']['r']}")
                        if self.args.tensorboard:
                            self.writer.add_scalar("charts/episodic_return",
                                                item["episode"]["r"], self.global_step)
                            self.writer.add_scalar("charts/episodic_length",
                                                item["episode"]["l"], self.global_step)

        # bootstrap value if not done
        with torch.no_grad():
            _, _, _, next_value = self.agent.get_action_and_value(
                next_obs, next_mask, action=None)

            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]

                delta = self.rewards[t] + self.args.gamma * \
                    nextvalues.sum(dim=-1) * nextnonterminal - \
                    self.values[t].sum(dim=-1)

                self.advantages[t] = lastgaelam = delta + self.args.gamma * \
                    self.args.gae_lambda * nextnonterminal * lastgaelam
                    
            self.returns = self.advantages + self.values.sum(dim=-1)

        # update env states
        self.next_obs = next_obs
        self.next_done = next_done
        self.next_mask = next_mask

    def pg_loss(self, advantages, ratio):

        # Policy loss

        pg_loss1 = -advantages * ratio

        pg_loss2 = -advantages * \
            torch.clamp(ratio, 1 - self.args.clip_coef,
                        1 + self.args.clip_coef)

        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        return pg_loss

    def v_loss(self, newvalue, returns, values):
        
        if self.args.clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = values + torch.clamp(
                newvalue - values,
                -self.args.clip_coef,
                self.args.clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(
                v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * \
                ((newvalue - returns) ** 2).mean()
        
        return v_loss

    def update(self):
        # flatten the batch
        b_obs = self.obs.reshape(
            (-1,) + self.envs.single_observation_space.shape)
        b_masks = self.masks.reshape((-1,) + (self.args.agents, 7))
        b_logprobs = self.logprobs.reshape(-1, self.args.agents)
        b_actions = self.actions.reshape(
            (-1,) + self.envs.single_action_space.shape)
        b_advantages = self.advantages.reshape(-1, 1)
        b_returns = self.returns.reshape(-1, 1)
        b_values = self.values.reshape(-1, self.args.agents)
        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)

        # p = 0.1 + (b_returns - b_returns.min()) / \
        #     (b_returns.max() - b_returns.min()) * 0.8
        # p = p / p.sum()
        # p = p.squeeze().cpu().numpy()
        
        clipfracs = []
        for epoch in range(self.args.update_epochs):
            # np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]
                
                # mb_inds = np.random.choice(
                #     b_inds, size=self.args.minibatch_size, replace=True, p=p)

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_masks[mb_inds], b_actions.long()[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                logratio = logratio.sum(-1, keepdim=True)

                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                if self.args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss = self.pg_loss(mb_advantages, ratio)

                v_loss = self.v_loss(
                    newvalue.sum(dim=-1), b_returns[mb_inds], b_values[mb_inds].sum(dim=-1))

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None:
                if approx_kl > self.args.target_kl:
                    break

        y_pred, y_true = b_values.sum(
            dim=-1).cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        if self.args.tensorboard:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate",
                                self.optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.add_scalar("losses/value_loss",
                                v_loss.item(), self.global_step)
            self.writer.add_scalar("losses/policy_loss",
                                pg_loss.item(), self.global_step)
            self.writer.add_scalar("losses/entropy",
                                entropy_loss.item(), self.global_step)
            self.writer.add_scalar("losses/old_approx_kl",
                                old_approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/approx_kl",
                                approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/clipfrac",
                                np.mean(clipfracs), self.global_step)
            self.writer.add_scalar("losses/explained_variance",
                                explained_var, self.global_step)
            print("SPS:", int(self.global_step / (time.time() - self.start_time)))
            self.writer.add_scalar("charts/SPS", int(self.global_step /
                                                    (time.time() - self.start_time)), self.global_step)

    def train(self):

        # TRY NOT TO MODIFY: start the game
        self.global_step = 0
        self.start_time = time.time()

        print('Beginning training...')
        self.next_obs, info = self.envs.reset(seed=self.args.seed)
        print('Finished resetting envs')
        self.next_obs = torch.Tensor(self.next_obs).to(self.device)
        self.next_done = torch.zeros(self.args.num_envs).to(self.device)
        self.next_mask = np.stack(info['mask'], axis=0)
        self.next_mask = torch.tensor(self.next_mask).to(self.device)

        for update in range(1, self.args.num_updates + 1):

            # Collect new data and save it in the global buffer
            self.collect(steps=self.args.num_steps)

            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / self.args.num_updates
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            self.update()

            if update % self.args.save_interval == 0:
                self.save()
                # self.play_trajectory(plays=2, use_keyboard=False)

        self.envs.close()
        self.writer.close()

    def play_trajectory(self, plays=2, use_keyboard=False):

        env = self.test_envs

        for play in range(plays):
            print('====================')
            print('====================')

            obs, info = env.reset()
            obs = torch.tensor(obs).to(self.device).unsqueeze(0)
            mask = torch.tensor(np.stack(info['mask'], axis=0)).to(
                self.device).unsqueeze(0)

            env.render()

            returns = []
            done = False
            while not done:
                print('------')
                print(obs, mask)

                if use_keyboard:
                    action = int(input('Action: '))
                    action = torch.tensor(action).to(
                        self.device).reshape(1, -1)
                    _, log_prob, entropy, value = self.agent.get_action_and_value(
                        obs, mask, action=action)
                else:
                    action, log_prob, entropy, value = self.agent.get_action_and_value(
                        obs, mask)

                action, log_prob, entropy, value = action.squeeze(
                    0), log_prob.squeeze(0), entropy.squeeze(0), value.squeeze(0)

                print('Action: ', action, 'Log Prob: ', log_prob,
                      'Entropy: ', entropy, 'Value: ', value)

                print(action, action.cpu().numpy())
                obs, reward, done, _, info = env.step(action.cpu().numpy())

                print('Reward: ', reward, 'Done: ', done)

                obs = torch.tensor(obs).to(self.device).unsqueeze(0)
                mask = torch.tensor(
                    np.stack(info['mask'], axis=0)).to(self.device).unsqueeze(0)

                env.render()

                returns += reward

            returns = np.array(returns)

    def save(self):
        save_dir = f"runs/{self.run_name}/models/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.agent.state_dict(),
                   save_dir + f"{self.global_step}.pt")

    def load(self, path):
        self.agent.load_state_dict(torch.load(path, map_location=self.device))

    def eval(self):
        raise NotImplementedError
