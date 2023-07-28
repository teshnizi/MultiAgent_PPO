import numpy as np


class MCTSNode:
    def __init__(self, state, reward, done, parent=None):
        self.state = state
        self.reward = reward
        self.done = done
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0


class MCTS:
    def __init__(self, agent, get_next, num_simulations, exploration_constant):
        self.agent = agent
        self.get_next = get_next
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant

    def search(self, root_state):
        root = MCTSNode(root_state, reward=0, done=False)

        for _ in range(self.num_simulations):
            node = self.select(root)
            if node.done:
                reward = node.reward
            else:
                node = self.expand(node)
                reward = self.simulate(node)
            self.backpropagate(node, reward)

        best_action, best_child = self.best_child(root)
        return best_action  # Return the action leading to the best child


    def select(self, node):
        while node.children:
            action, node = self.best_uct(node)
        return node

    def expand(self, node):
        action_probs = self.agent.get_action_probs(node.state)
        for action, prob in enumerate(action_probs):
            next_state, reward, done = self.get_next(node.state, action)
            child = MCTSNode(next_state, reward, done, parent=node)
            node.children[action] = child
        return node
    
    def simulate(self, node):
        state = node.state
        done = node.done
        total_reward = 0
        while not done:
            action_probs = self.agent.get_action_probs(state)
            action = np.random.choice(len(action_probs), p=action_probs)
            next_state, reward, done = self.get_next(state, action)
            total_reward += reward
            state = next_state
        return total_reward

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def best_child(self, node):
        best_action, best_child = max(node.children.items(),
                                      key=lambda item: item[1].total_reward / item[1].visits if item[1].visits > 0 else 0)
        return best_action, best_child


    def best_uct(self, node):
        uct_values = [(action, child) for action, child in node.children.items()]
        action, child = max(uct_values, key=lambda item: self.uct_value(item[1]))
        
        return action, child

    def uct_value(self, node):
        if node.visits == 0:
            return np.inf  # Infinite UCT value for unvisited nodes
        else:
            exploit = node.total_reward / node.visits
            explore = np.sqrt(2 * np.log(node.parent.visits) / node.visits)
            return exploit + self.exploration_constant * explore




class DummyAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def get_action_probs(self, state):
        return np.ones(self.num_actions) / self.num_actions  # Uniformly random actions
    

def dummy_get_next(state, action):
    if state == 0:  # Start state
        if action == 0:  # Wrong action
            return 0, -1, False
        elif action == 1:  # Correct action
            return 1, 1, True
    else:  # Terminal state
        return state, 0, True

agent = DummyAgent(num_actions=2)
mcts = MCTS(agent, dummy_get_next, num_simulations=20, exploration_constant=1)

lst = []
for i in range(2000):
    action = mcts.search(root_state=0) # Should print 1 with high probability
    lst.append(action)
    
print(sum(lst)/len(lst))
