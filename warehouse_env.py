import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# things to add:
# - agent speeds
# - agent capacities
# - agent charging & stations
# - object weights
# - agent-object constraints
# - object-object constraints
# - relative agent-object speeds
# - privacy constraints
# - multiple warehouses and inter-warehouse delivery Agents

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class WarehouseEnv(gym.Env):
    """Warehouse environment

        Parameters
        ----------
        N : int
            Grid dimension
        Agents: int
            Number of agents
        objects: int
            Number of objects to be delivered

        Actions
        -------
        Type: MultiDiscrete(N)
        0: Do Nothing
        1: Move Right
        2: Move Up
        3: Move Left
        4: Move Down
        5: Pick Up
        6: Drop Off
    """

    def __init__(self, grid_size=6, n_agents=2, n_objects=3, warmup=True, seed=None):
        super(WarehouseEnv, self).__init__()

        # Set the seed

        np.random.seed(seed)
        random.seed(seed)

        self.N = grid_size  # grid size
        self.agent_num = n_agents  # number of agents
        self.object_num = n_objects  # number of objects

        self.action_space = spaces.MultiDiscrete(
            [7]*self.agent_num)  # one action for each node

        # For agent rows, values are (x, y, object, -1)
        # For object rows, values are (x, y, x_dest, y_dest)
        self.observation_space = spaces.Box(
            low=-1, high=self.N, shape=(self.agent_num + self.object_num, 4), dtype=np.float32)

        self.max_steps = (self.N * 4 * self.object_num)//self.agent_num
        self.warmup = warmup

    def step(self, action):

        # print('----')
        # print('Actions: ', action)
        # if action is numpy int64, convert to numpy array:
        if isinstance(action, np.int64):
            action = np.array([action])

        assert self.action_space.contains(action), "Invalid action!"

        # assert all agents are within bounds
        assert np.all(self.agents[:, 0:2] >= 0) and np.all(
            self.agents[:, 0:2] < self.N), "Agent out of bounds!"

        # assert all objects are within bounds
        # assert np.all(self.objects[:, 0:2] >= 0) and np.all(
        #     self.objects[:, 0:2] < self.N), f"Object out of bounds! {self.objects}"

        # assert all agents are not on top of each other
        # assert np.unique(
        #     self.agents[:, 0:2], axis=0).shape[0] == self.agents.shape[0], "Agents on top of each other!"

        # print('Agents: ', self.agents)
        # print('Objects: ', self.objects)
        # print('Agent of Object: ', self.agent_of_object)

        # make sure no two objects re taken by the same agent (no repetitive positive entries in self.agent_of_object)
        assert np.unique(self.agent_of_object[self.agent_of_object >= 0]).shape[0] == self.agent_of_object[
            self.agent_of_object >= 0].shape[0], "Two agents are carrying the same object!"

        # make sure no two objects are taken to the same destination (no repetitive entries in self.agents[:, 3])

        assert np.unique(self.agents[:, 3][self.agents[:, 3] >= 0]).shape[0] == self.agents[:,
                                                                                            3][self.agents[:, 3] >= 0].shape[0], "Two objects are taken to the same destination!"

        rewards = np.zeros(self.agent_num)

        for a in range(self.agent_num):

            if action[a] == 0:  # Do nothing
                pass

            elif action[a] == 1:  # Move Right
                if not self.agents[a, 0] < self.N - 1:
                    rewards[a] -= 5
                    continue
                self.agents[a, 0] += 1
                if self.agents[a, 3] != -1:
                    self.objects[int(self.agents[a, 3]), 0] += 1

            elif action[a] == 2:  # Move Up
                if not self.agents[a, 1] < self.N - 1:
                    rewards[a] -= 5
                    continue
                self.agents[a, 1] += 1
                if self.agents[a, 3] != -1:
                    self.objects[int(self.agents[a, 3]), 1] += 1

            elif action[a] == 3:  # Move Left
                if not self.agents[a, 0] > 0:
                    rewards[a] -= 5
                    continue
                self.agents[a, 0] -= 1
                if self.agents[a, 3] != -1:
                    self.objects[int(self.agents[a, 3]), 0] -= 1

            elif action[a] == 4:  # Move Down
                if not self.agents[a, 1] > 0:
                    rewards[a] -= 5
                    continue

                self.agents[a, 1] -= 1
                if self.agents[a, 3] != -1:
                    self.objects[int(self.agents[a, 3]), 1] -= 1

            elif action[a] == 5:  # Pick Up

                untaken_objects = np.where(self.agent_of_object == -1)[0]
                objects_at_agent = np.where(
                    (self.objects[:, 0:2] == self.agents[a, 0:2]).all(axis=1))[0]

                # print('eq ', self.objects[:, 0:2] == self.agents[a, 0:2])
                # print('oaa ', objects_at_agent)

                valid_objects = np.intersect1d(
                    untaken_objects, objects_at_agent)

                # print(valid_objects)
                # make sure the object is not already taken
                if self.agent_of_object[valid_objects] != -1:
                    continue

                # assert valid_objects.size == 1, "0 or Multiple objects at agent's location!"
                if valid_objects.size == 0:
                    rewards[a] -= 5
                    continue
                self.agents[a, 3] = valid_objects[0]

                self.agent_of_object[valid_objects[0]] = a
                # add a reward if the object is never picked up before
                if self.virgin_objects[valid_objects[0]]:
                    rewards[a] += self.N
                    self.virgin_objects[valid_objects[0]] = False

            elif action[a] == 6:  # Drop Off
                # assert self.agents[a, 3] != -1, "Agent not carrying an object!"
                if not self.agents[a, 3] == -1:
                    rewards[a] -= 5

                self.agent_of_object[int(self.agents[a, 3])] = -1
                self.agents[a, 3] = -1

                rewards[a] -= 2

            if self.agents[a, 3] != -1:
                if (self.objects[int(self.agents[a, 3]), 0:2] == self.objects[int(self.agents[a, 3]), 2:4]).all():
                    rewards[a] += self.N * 3

                    self.agent_of_object[int(self.agents[a, 3])] = -1
                    self.objects[int(self.agents[a, 3]), :] = -4

                    self.agents[a, 3] = -1

        reward = np.sum(rewards)

        # for obj in range(self.object_num):
        #     if self.agent_of_object[obj] > -1

        # subtract distances of the objects from their destinations from the reward
        reward -= np.sum(np.abs(self.objects[:,
                         0:2] - self.objects[:, 2:4]))/(self.N * self.object_num)

        # subtract distances of the agents from the first object from the reward

        # reward -= np.sum(
        #     np.abs(self.agents[:, 0:2] - self.objects[0, 0:2]))

        self.current_step += 1

        # environment terminates when all objects are at their destinations or when max_steps is reached
        done = (self.objects[:, 0] <= -
                1).all() or self.current_step >= self.max_steps

        obs = np.concatenate((self.agents, self.objects), axis=0)

        info = {'mask': self.calculate_mask(obs)}

        return obs, reward, done, False, info

    def calculate_mask(self, obs):
        """ Calculate action mask for each agent

        Parameters
        ----------
        obs : np.array (Agents + objects, 4)
            Observation array

        Returns
        -------
        mask : np.array (Agents, 7)
            Action mask
        """

        # Initialize mask
        mask = np.zeros((self.agent_num, 7))

        agent_data = obs[:self.agent_num, :]
        object_data = obs[self.agent_num:, :]

        # Calculate mask for each agent
        for i in range(self.agent_num):

            # Do Nothing
            mask[i, 0] = 1

            # Move Right if not at edge and not blocked by another agent
            if agent_data[i, 0] < self.N - 1 and not np.any(np.all(agent_data[:, 0:2] == agent_data[i, 0:2] + np.array([1, 0]), axis=1)):
                mask[i, 1] = 1

            # Move Up if not at edge and not blocked by another agent
            if agent_data[i, 1] < self.N - 1 and not np.any(np.all(agent_data[:, 0:2] == agent_data[i, 0:2] + np.array([0, 1]), axis=1)):
                mask[i, 2] = 1

            # Move Left if not at edge and not blocked by another agent
            if agent_data[i, 0] > 0 and not np.any(np.all(agent_data[:, 0:2] == agent_data[i, 0:2] + np.array([-1, 0]), axis=1)):
                mask[i, 3] = 1

            # Move Down if not at edge and not blocked by another agent
            if agent_data[i, 1] > 0 and not np.any(np.all(agent_data[:, 0:2] == agent_data[i, 0:2] + np.array([0, -1]), axis=1)):
                mask[i, 4] = 1

            # Pick Up if object is at agent location and agent is not already carrying an object
            if agent_data[i, 3] == -1 and np.any(np.all(object_data[:, 0:2] == agent_data[i, 0:2], axis=1)):
                obj = np.where(
                    np.all(object_data[:, 0:2] == agent_data[i, 0:2], axis=1))[0][0]
                if self.agent_of_object[obj] == -1:
                    mask[i, 5] = 1

            # Drop Off if there's no other object at agent location and agent is carrying an object
            if (agent_data[i, 3] != -1) and (np.sum(np.all(object_data[:, 0:2] == agent_data[i, 0:2], axis=1)) == 1):
                mask[i, 6] = 1

        # turn to boolean
        mask = mask.astype(bool)

        return mask

    def reset(self, seed=None, options=None):

        # Set the seed
        if seed is not None:
            np.random.seed(seed)

        
        self.objects = np.zeros((self.object_num, 4), dtype=np.float32) - 1
        self.agents = np.zeros((self.agent_num, 4), dtype=np.float32) - 1

        self.virgin_objects = np.zeros((self.object_num, 1)) < 1

        self.agent_of_object = np.zeros((self.object_num, 1)) - 1

        # sample locations and destinations for each object without replacement
        tmp = np.random.choice(self.N**2, size=self.object_num,
                                 replace=False).reshape(self.object_num)
        x, y = tmp // self.N, tmp % self.N
        self.objects[:, 0:2] = np.stack((x, y), axis=1)
        
        tmp = np.random.choice(self.N**2, size=self.object_num,
                                    replace=True).reshape(self.object_num)
        
        x, y = tmp // self.N, tmp % self.N
        self.objects[:, 2:4] = np.stack((x, y), axis=1)

        # sample locations for each agent without replacement
        tmp = np.random.choice(self.N**2, size=self.agent_num,
                                    replace=False).reshape(self.agent_num)
        x, y = tmp // self.N, tmp % self.N
        self.agents[:, 0:2] = np.stack((x, y), axis=1)

        self.current_step = 0

        obs = np.concatenate((self.agents, self.objects), axis=0)
        info = {'mask': self.calculate_mask(obs)}

        return obs, info

    def render(self, mode='human'):
        fig, ax = plt.subplots()
        ax.set_xlim(-1, self.N)
        ax.set_ylim(-1, self.N)
        # add N by N grid
        ax.set_xticks(np.arange(0, self.N, 1))
        ax.set_yticks(np.arange(0, self.N, 1))
        ax.grid()

        # draw agents as red dots
        for a in self.agents:
            ax.add_patch(Rectangle((a[0]-0.5, a[1]-0.5), 1, 1, color='r'))

        # draw objects as blue dots, add a border if they are being carried
        for o in self.objects:
            if o[3] == -1:
                ax.add_patch(Circle((o[0], o[1]), 0.5, color='b'))
            else:
                ax.add_patch(Circle((o[0], o[1]), 0.5, color='b', fill=False))

            # also show destination of objects
            ax.add_patch(Circle((o[2], o[3]), 0.5, color='g'))
        plt.show()


# register the env
gym.register(id='Warehouse-v0',
             entry_point='warehouse_env:WarehouseEnv')

