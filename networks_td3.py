import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorNetwork(nn.Module):
    """
    Actor network for single GPU. 
    """
    def __init__(self, env_id, input_dims, fc1_dim, fc2_dim, num_actions, max_action, lr_alpha, 
            algo_name, loss_type, nn_name):
        """
        Intialise class varaibles by creating neural network with Adam optimiser.

        Parameters:
            env_id (string): name of gym environment
            input_dim (list): dimensions of inputs
            fc1_dim (int): size of first fully connected layer
            fc2_dim (int): size of second fully connected layer
            num_actions (int): number of actions available to the agent
            max_action (float): maximium possible value of action in environment
            lr_alpha (float): actor learning rate of Adam optimiser
            algo_name (string): name of algorithm
            loss_type (str): Cauchy, CE, Huber, KL, MAE, MSE, TCauchy loss functions
            nn_name (string): name of network
        """
        super(ActorNetwork, self).__init__()
        self.env_id = str(env_id)
        self.input_dims = input_dims
        self.fc1_dim = int(fc1_dim)
        self.fc2_dim = int(fc2_dim)
        self.num_actions = int(num_actions)
        self.max_action = float(max_action)
        self.lr_alpha = lr_alpha
        self.algo_name = str(algo_name)
        self.loss_type = str(loss_type)
        self.nn_name = str(nn_name)
        
        # directory to save network checkpoints
        if not os.path.exists('./models/'+'/'+env_id):
            os.makedirs('./models/'+'/'+env_id)
        self.file_checkpoint = os.path.join('./models/'+'/'+env_id, self.env_id+'--'+self.algo_name
                                        +'_'+self.loss_type+'_'+self.nn_name)

        # network inputs environment space shape
        self.fc1 = nn.Linear(sum(self.input_dims), self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.mu = nn.Linear(self.fc2_dim, self.num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        """
        Forward propogation of state to obtain actor psuedo-probabilistic action.

        Parameters:
            state (list): current environment state

        Returns:
            prob (float): rank agent actions between -1 and 1 scaled by max action
        """
        action_prob = self.fc1(state)
        action_prob = F.relu(action_prob)
        action_prob = self.fc2(action_prob)
        action_prob = F.relu(action_prob)
        prob = T.tanh(self.mu(action_prob)) * self.max_action

        return prob

    def save_checkpoint(self):
        # print('... saving checkpoint')
        T.save(self.state_dict(), self.file_checkpoint)

    def load_checkpoint(self):
        # print('... loading checkpoint')
        self.load_state_dict(T.load(self.file_checkpoint))

class CriticNetwork(nn.Module):
    """
    Critic network for single GPU. 
    """
    def __init__(self, env_id, input_dims, fc1_dim, fc2_dim, num_actions, max_action, lr_beta, 
            algo_name, loss_type, nn_name):
        """
        Intialise class varaibles by creating neural network with Adam optimiser.

        Parameters:
            env_id (string): name of gym environment
            input_dim (list): dimensions of inputs
            fc1_dim (int): size of first fully connected layer
            fc2_dim (int): size of second fully connected layer
            num_actions (int): number of actions available to the agent
            max_action (float): maximium possible value of action in environment
            lr_beta (float): critic learning rate of Adam optimiser
            nn_name (string): name of network
            loss_type (str): Cauchy, CE, Huber, KL, MAE, MSE, TCauchy loss functions
            algo_name (string): name of algorithm
        """
        super(CriticNetwork, self).__init__()
        self.env_id = str(env_id)
        self.input_dims = input_dims
        self.fc1_dim = int(fc1_dim)
        self.fc2_dim = int(fc2_dim)
        self.num_actions = int(num_actions)
        self.max_action = float(max_action)
        self.lr_beta = lr_beta
        self.algo_name = str(algo_name)
        self.loss_type = str(loss_type)
        self.nn_name = str(nn_name)
        
        # directory to save network checkpoints
        if not os.path.exists('./models/'+'/'+env_id):
            os.makedirs('./models/'+'/'+env_id)
        self.file_checkpoint = os.path.join('./models/'+'/'+env_id, self.env_id+'--'+self.algo_name
                                        +'_'+self.loss_type+'_'+self.nn_name)

        # network inputs environment space shape and number of actions
        self.fc1 = nn.Linear(sum(self.input_dims) + self.num_actions, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.q = nn.Linear(self.fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        """
        Forward propogation of state-action pair to obtain Q-value.

        Parameters:
            state (list): current environment state
            action (list): continuous actions taken to arrive at current state          

        Returns:
            Q (float): estimated Q action-value
        """
        Q_action_value = self.fc1(T.cat([state, action], dim=1))
        Q_action_value = F.relu(Q_action_value)
        Q_action_value = self.fc2(Q_action_value)
        Q_action_value = F.relu(Q_action_value)
        Q = self.q(Q_action_value)

        return Q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.file_checkpoint)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.file_checkpoint))
