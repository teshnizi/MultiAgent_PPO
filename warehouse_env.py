import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

import pygame

import utils

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

import time


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
            low=-1, high=self.N, shape=(self.agent_num + self.object_num, 6), dtype=np.float32)

        self.max_steps = (self.N * 4 * self.object_num)//self.agent_num
        # self.warmup = warmup


    def step_logic(self, state, action):
        
        if isinstance(action, np.int64):
            action = np.array([action])
         
        agents = state[:self.agent_num]
        objects = state[self.agent_num:]
        
        assert self.action_space.contains(action), "Invalid action!"

        # assert all agents are within bounds
        assert np.all(agents[:, 0:2] >= 0) and np.all(
            agents[:, 0:2] < self.N), "Agent out of bounds!"

        # make sure no two objects re taken by the same agent (no repetitive positive entries in self.objects[:, 4])
        assert np.unique(objects[objects[:, 4] >= 0, 4]).shape[0] == objects[
            objects[:, 4] >= 0, 4].shape[0], "Two agents are carrying the same object!"

        # make sure no two objects are taken to the same destination (no repetitive entries in agents[:, 3])

        assert np.unique(agents[:, 3][agents[:, 3] >= 0]).shape[0] == agents[:,
                                                                                            3][agents[:, 3] >= 0].shape[0], "Two objects are taken to the same destination!"

        rewards = np.zeros(self.agent_num)
        
        for a in range(self.agent_num):

            if action[a] == 0:  # Do nothing
                pass

            elif action[a] == 1:  # Move Right
                agents[a, 0] += 1
                if agents[a, 3] != -1:
                    objects[int(agents[a, 3]), 0] += 1

            elif action[a] == 2:  # Move Up
                agents[a, 1] += 1
                if agents[a, 3] != -1:
                    objects[int(agents[a, 3]), 1] += 1

            elif action[a] == 3:  # Move Left
                agents[a, 0] -= 1
                if agents[a, 3] != -1:
                    objects[int(agents[a, 3]), 0] -= 1

            elif action[a] == 4:  # Move Down
                agents[a, 1] -= 1
                if agents[a, 3] != -1:
                    objects[int(agents[a, 3]), 1] -= 1

            elif action[a] == 5:  # Pick Up
                untaken_objects = np.where(objects[:, 4] == -1)[0]
                objects_at_agent = np.where(
                    (objects[:, 0:2] == agents[a, 0:2]).all(axis=1))[0]

                valid_objects = np.intersect1d(
                    untaken_objects, objects_at_agent)

                # assert valid_objects.size >= 1, f"0 objects at agent's location! \n Agents: {agents} \n Objects: {objects} \n Agent of Object: {self.agent_of_object}, \n valid_objects: {valid_objects}, objects_at_agent: {objects_at_agent}"
                
                if len(valid_objects) == 0:
                    continue # object taken by another agent in the same action

                # choose the first valid object that is not taken
                for i in range(valid_objects.shape[0]):
                    if objects[valid_objects[i], 4] == -1:
                        break
                    
                valid_object = valid_objects[i]

                agents[a, 3] = valid_object

                objects[valid_object, 4] = a

                # # add a reward if the object is never picked up before
                if objects[valid_object, 5] == 1:
                    rewards[a] += 10 / self.object_num
                    objects[valid_object, 5] = 0

            elif action[a] == 6:  # Drop Off
                assert agents[a, 3] != -1, "Agent not carrying an object!"

                objects[int(agents[a, 3]), 4] = -1
                agents[a, 3] = -1
                


            if agents[a, 3] != -1:
                if (objects[int(agents[a, 3]), 0:2] == objects[int(agents[a, 3]), 2:4]).all():
                    rewards[a] += 40 / self.object_num

                    objects[int(agents[a, 3]), 4] = -1
                    objects[int(agents[a, 3]), :] = -4

                    agents[a, 3] = -1

        reward = np.sum(rewards)

        # subtract distances of the objects from their destinations from the reward
        # reward -= np.sum(np.abs(objects[:,
        #                  0:2] - objects[:, 2:4]))/(self.N * self.object_num)
        
        # environment terminates when all objects are at their destinations or when max_steps is reached
        done = (objects[:, 0] <= -
                1).all() or self.current_step >= self.max_steps
        
        if done:
            if (objects[:, 0] <= -1).all():
                reward += 50

        obs = np.concatenate((agents, objects), axis=0)
        mask = self.calculate_mask(obs)
        
        return obs, reward, done, mask
    
    
        
    def step(self, action):
        
        self.current_step += 1
        
        current_obs = np.concatenate((self.agents, self.objects), axis=0)
        new_obs, reward, done, new_mask = self.step_logic(current_obs, action)
        self.agents = new_obs[:self.agent_num]
        self.objects = new_obs[self.agent_num:]
        
        info = {'mask': new_mask}
        return new_obs, reward, done, False, info
    

    def calculate_mask(self, obs):
        """ Calculate action mask for each agent

        Parameters
        ----------
        obs : np.array (Agents + objects, 5)
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
                if self.objects[obj, 4] == -1:
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

        # features for each object: x, y, dx, dy, agent_id, virginity
        # features for each agent: x, y, -1, object_id, -1, -1
        
        self.objects = np.zeros((self.object_num, 6), dtype=np.float32) - 1
        self.agents = np.zeros((self.agent_num, 6), dtype=np.float32) - 1
        

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

        self.objects[:, 5] = 1
        
        obs = np.concatenate((self.agents, self.objects), axis=0)
        info = {'mask': self.calculate_mask(obs)}
        
        return obs, info
    

    def render(self, mode='human', close=False):
        block_size = 64

        # Initialize pygame if it hasn't been done yet
        if not pygame.get_init():
            pygame.init()

            self.screen = pygame.display.set_mode(
                (self.N*block_size, self.N*block_size))

        # Fill the background with white
        self.screen.fill((255, 255, 255))

        # # Draw the grid
        for x in range(0, self.N*block_size, block_size):
            for y in range(0, self.N*block_size, block_size):
                pygame.draw.rect(self.screen, (0, 0, 0),
                                 pygame.Rect(x, y, block_size, block_size), 1)

        for i, o in enumerate(self.objects):

            utils.draw_object(
                self.screen, o[0]*block_size, o[1] * block_size, block_size, is_taken=self.objects[i, 4] != -1)

            utils.draw_destination(
                self.screen, o[2]*block_size, o[3]*block_size, block_size)

        for a in self.agents:

            print(a[0], a[1])
            utils.draw_agent(
                self.screen, a[0]*block_size, a[1] * block_size, block_size)

        # Update the display
        pygame.display.flip()
        pygame.event.get()

        pygame.time.delay(150)


    # a function that takes in a state and an action and returns the next state, reward, and done
    def simulate(state, action):
        pass
        
        
        
# register the env
gym.register(id='Warehouse-v0',
             entry_point='warehouse_env:WarehouseEnv')
