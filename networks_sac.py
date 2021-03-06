import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distibution

class ActorNetwork(nn.Module):
    """
    Actor network for single GPU. 
    """
    def __init__(self, env_id, input_dims, fc1_dim, fc2_dim, num_actions, max_action, lr_alpha, 
            reparam_noise, algo_name, loss_type, nn_name):
        """
        Intialise class varaibles by creating neural network with Adam optimiser.

        Parameters:
            env_id (string): name of gym environment
            input_dim (list): dimensions of inputs
            fc1_dim (int): size of first fully connected layer
            fc2_dim (int): size of second fully connected layer
            num_actions (int): number of actions available to the agent
            max_action (float): maximium possible value of action in environment
            lr_alpha (float>0): actor learning rate of Adam optimiser
            reparam_noise (float>0): miniscule constant for valid logarithm
            algo_name (string): name of algorithm
            loss_type (str): critic loss functions
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
        self.reparam_noise = reparam_noise
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
        self.mu = nn.Linear(self.fc2_dim, self.num_actions * 2)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        """
        Forward propogation of state to obtain fixed Gaussian distribution parameters
        (moments) for each possible action component. 

        Parameters:
            state (list): current environment state

        Returns:
            moments (float): first half columns for deterministic loc of action components
                             and second half columns for log scales of action components
        """
        actions_2x = self.fc1(state)
        actions_2x = F.relu(actions_2x)
        actions_2x = self.fc2(actions_2x)
        actions_2x = F.relu(actions_2x)
        moments = self.mu(actions_2x)

        return moments
    
    def stochastic_uv(self, state, stoch='N'):
        """ 
        Stochastic action selection sampled from several unbounded univarite distirbution
        using the reparameterisation trick from https://arxiv.org/pdf/1312.6114.pdf. Addition
        of constant reparameterisation noise to the logarithm is crucial, as verified in both
        https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC.py
        https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/blob/master/SAC/networks.py
        where orders of magnitude smaller than 1e-6 prevent learning from occuring. 
        
        Parameters:
            state (list): current environment state or mini-batch
            stoch (str): stochastic policy 'L' or 'N' distribution

        Returns:
            bounded_action (list, float): action truncated by tanh and scaled by max action
            bounded_logprob_action (float): log probability of sampled truncated action 
        """
        moments = self.forward(state)
        mu, log_scale = moments[:, :self.num_actions], moments[:, self.num_actions:]
        scale = log_scale.exp()
        
        if stoch == 'N':
            probabilities = distibution.normal.Normal(loc=mu, scale=scale)
        else:
            probabilities = distibution.laplace.Laplace(loc=mu, scale=scale)

        # reparmeterise trick for random variable sample to be pathwise differentiable
        unbounded_action = probabilities.rsample().to(self.device)
        bounded_action = T.tanh(unbounded_action) * self.max_action
        unbounded_logprob_action = probabilities.log_prob(unbounded_action).sum(1, keepdim=True).to(self.device)
        
        # ensure defined bounded log by adding minute noise
        log_inv_jacobian = T.log(1 - (bounded_action / self.max_action)**2 + self.reparam_noise).sum(dim=1, keepdim=True)
        bounded_logprob_action = unbounded_logprob_action - log_inv_jacobian

        return bounded_action, bounded_logprob_action
    
    def stochastic_mv_gaussian(self, state):
        """
        Stochastic action selection sampled from unbounded spherical Gaussian input noise 
        with tanh bounding using Jacobian transformation.

        Parameters:
            state (list): current environment state or mini-batch

        Returns:
            bounded_action (list, float): action truncated by tanh and scaled by max action
            bounded_logprob_action (float): log probability of sampled truncated action 
        """
        moments = self.forward(state)
        batch_size = moments.size()[0]
        print(moments)
        mu, log_var = moments[:, :self.num_actions], moments[:, self.num_actions:]
        var = log_var.exp()

        if batch_size > 1:
            pass
        else:
            mu, var = mu.view(-1), var.view(-1)
        
        # create diagonal covariance matrices for each sample and perform Cholesky decomposition
        cov_mat = T.stack([T.eye(self.num_actions) for i in range(batch_size)]).to(self.device)

        if batch_size > 1:
            for sample in range(batch_size):
                for vol in range(self.num_actions):
                        cov_mat[sample, vol, vol] = var[sample, vol]    # diagonal elements are variance
        else:
            for vol in range(self.num_actions):     
                cov_mat[0, vol, vol] = var[vol]

        chol_ltm = T.cholesky(cov_mat)

        probabilities = distibution.multivariate_normal.MultivariateNormal(loc=mu, scale_tril=chol_ltm)

        # reparmeterise trick for random variable sample to be pathwise differentiable
        unbounded_action = probabilities.rsample().to(self.device)
        bounded_action = T.tanh(unbounded_action) * self.max_action
        unbounded_logprob_action = probabilities.log_prob(unbounded_action).to(self.device)

        # ensure defined bounded log by adding minute noise
        log_inv_jacobian = T.log(1 - (bounded_action / self.max_action)**2 + self.reparam_noise).sum(dim=1)
        bounded_logprob_action = unbounded_logprob_action - log_inv_jacobian

        return bounded_action, bounded_logprob_action

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
            lr_beta (float>0): critic learning rate of Adam optimiser
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
        Forward propogation of state-action pair to obtain soft Q-value.

        Parameters:
            state (list): current environment state
            action (list): continuous actions taken to arrive at current state          

        Returns:
            soft_Q (float): estimated soft Q action-value
        """
        Q_action_value = self.fc1(T.cat([state, action], dim=1))
        Q_action_value = F.relu(Q_action_value)
        Q_action_value = self.fc2(Q_action_value)
        Q_action_value = F.relu(Q_action_value)
        soft_Q = self.q(Q_action_value)

        return soft_Q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.file_checkpoint)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.file_checkpoint))
