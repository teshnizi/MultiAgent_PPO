import numpy as np
import torch
import torch.nn as nn

from time import time


class MCTSNode:
    def __init__(self, state, mask, agent, reward, value, done, prob, parent=None):
        self.state = state
        self.mask = mask
        self.agent = agent
        self.visits = 0
        
        self.reward = reward
        self.value = value # value of this node after coming from its parent
        self.prob = prob # probability of visiting this node from the parent
        self.done = done
        
        
        self.Q = 0
        self.parent = parent
        self.children = {}
        self.action_probs = None
        
        # self.total_reward = 0

        
    # define what happens when you print the node
    def __repr__(self):
        # return f"MCTSNode(state={self.state}, mask={self.mask}, reward={self.reward}, done={self.done}, parent={self.parent.state if self.parent != None else None},\n visits={self.visits}"
        return f"MCTSNode (state={self.state.shape})"


class MCTS(nn.Module):
    def __init__(self, model, env, args):
        super().__init__()
        self.model = model
        self.env = env
        self.num_simulations = args.num_simulations
        self.exploration_constant = args.exploration_constant
        self.args = args

    def search(self, root_state, root_mask):
        
        roots = []
        root_state = root_state.cpu().numpy()
        root_mask = root_mask.cpu().numpy()
        
        for env_id in range(root_state.shape[0]):
            for agent in range(self.args.agents):
                root = MCTSNode(root_state[env_id], root_mask[env_id], agent, reward=0, value=0, done=False, prob=1, parent=None)
                roots.append(root)
        
        for _ in range(self.num_simulations):
            
            nodes = [self.select(root) for root in roots]
            
            masks = torch.Tensor([node.mask for node in nodes]).to(self.model_device).bool()            
            states = torch.Tensor([node.state for node in roots]).to(self.model_device)
            
            with torch.no_grad():
                _, _, _, value, action_probs = self.model(states, masks)
            
            for i, node in enumerate(nodes):
                node.action_probs = action_probs[i].cpu().numpy()
                node.value = value[i].cpu().numpy()
                
                node = self.expand(node)
                self.backpropagate(node)
        
        best_action = np.zeros((root_state.shape[0], self.args.agents))
        
        for env_id in range(root_state.shape[0]):
            for agent in range(self.args.agents):
                best_action[env_id, agent] = self.best_child(roots[env_id * self.args.agents + agent])
        
        return best_action  # Return the action leading to the best child


    def select(self, node):
        while node.children:
            action, node = self.best_uct(node)
        return node

    def expand(self, node):

        # TODO: figure out the parallelization situation
        
        chosen_action = node.action_probs.argmax(axis=-1)
        for agent_action, prob in enumerate(node.action_probs[node.agent]):
            if node.mask[node.agent, agent_action] == 0:
                continue
            
            chosen_action[node.agent] = agent_action
            next_state, reward, done, next_mask = self.env.step_logic(node.state.copy(), chosen_action)
            child = MCTSNode(next_state, next_mask, node.agent, reward, value=None, done=done, prob=prob, parent=node)
            node.children[agent_action] = child
        
        return node


    def backpropagate(self, node):
        current_value = node.value.sum()
        while node:
            node.Q = (node.Q * node.visits + current_value) / (node.visits + 1)
            current_value = current_value * self.args.gamma + node.reward
            node.visits += 1
            node = node.parent

    def best_child(self, node):
        # best_action, best_child = max(node.children.items(), key=lambda item: item[1].N)
        prob_dist = np.array([child.visits for child in node.children.values()])
        
       
        # use softmax to get the probability distribution
        prob_dist = np.exp(prob_dist) / np.sum(np.exp(prob_dist))
    
        # sample from the distribution to get the action
        dist = torch.distributions.Categorical(torch.from_numpy(prob_dist))
        
        child_ids = np.array(list(node.children.keys()))
        
        action = dist.sample().item() 
        # action = dist.probs.argmax().item()
        
        return child_ids[action]


    def best_uct(self, node):
        children = [(action, child) for action, child in node.children.items()]
        uct_values = [(action, self.uct_value(child)) for action, child in children]
        
        action, uct_value = max(uct_values, key=lambda item: item[1])
        child = node.children[action]
        return action, child

    def uct_value(self, node):
        exploit = node.Q/100.0
        explore = node.prob * np.sqrt(node.parent.visits) / (1 + node.visits) * np.sqrt(2)
        
        return exploit + self.exploration_constant * explore

    def forward(self, state, mask, action):
        self.model_device = self.model.parameters().__next__().device
        
        if self.args.num_simulations > 0: # Use MCTS if num_simulations > 0
        
            if action is None:
                action = self.search(state.clone(), mask.clone())
                action = torch.from_numpy(action).to(self.model_device).long()
                
            # check actions are not masked
            assert (mask.gather(-1, action.unsqueeze(-1)) == 1).all()

        action, log_prob, entropy, value, probs = self.model(state, mask, action)
            
        return action, log_prob, entropy, value, probs
        
        
class DummyAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def forward(self, state):
        return np.ones(self.num_actions) / self.num_actions  # Uniformly random actions
    

def dummy_get_next(state, action):
    if state == 0:  # Start state
        if action == 0:  # Wrong action
            return 0, -1, False
        elif action == 1:  # Correct action
            return 1, 1, True
    else:  # Terminal state
        return state, 0, True

# agent = DummyAgent(num_actions=2)
# mcts = MCTS(agent, dummy_get_next, {}, num_simulations=20, exploration_constant=1)

# lst = []
# for i in range(2000):
#     action = mcts.search(root_state=0) # Should print 1 with high probability
#     lst.append(action)
    
# print(sum(lst)/len(lst))
