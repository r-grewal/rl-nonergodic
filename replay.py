import numpy as np

class ReplayBuffer():
    """
    Experience replay buffer with uniform sampling.
    """

    def __init__(self, max_size, input_shape, num_actions):
        """
        Intialise class varaibles by creating empty numpy arrays.

        Paramters:
            max_size (int): maximum size of replay buffer
            input_shape (list): dimensions of environment state
            num_actions (int): number of actions available to the agent
        """
        self.mem_size = int(max_size)
        self.mem_idx = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, num_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_exp(self, state, action, reward, next_state, done):
        """
        Store a transistion to the buffer containing a total up to max_size.

        Paramters:
            state (list): current environment state
            action (list): continuous actions taken to arrive at current state
            reward (float): reward from current environment state
            next_state (list): next environment state
            done (bool): flag if current state is terminal
        """
        idx = self.mem_idx % self.mem_size

        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.terminal_memory[idx] = done

        self.mem_idx += 1

    def sample_exp(self, batch_size):
        """
        Uniformly sample a batch from replay buffer for gradient descent.

        Paramters:
            batch_size (int): mini-batch size

        Returns:
            states (array): batch of environment states
            actions (array): batch of continuous actions taken to arrive at states
            rewards (array): batch of rewards from current states
            next_states (array): batch of next environment states
            dones (array): batch of done flags
        """
        # pool batch from either partial or fully populated buffer
        max_mem = min(self.mem_idx, self.mem_size)
        batch = np.random.choice(max_mem, size=batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones