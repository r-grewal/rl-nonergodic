import numpy as np
import torch as T
from replay import ReplayBuffer
from networks_sac import ActorNetwork, CriticNetwork
import utils 

class Agent_sac():
    """
    SAC agent algorithm.
    """                             
    def __init__(self, env_id, env, input_dims, num_actions, lr_alpha=3e-4, lr_beta=3e-4, lr_kappa=3e-4, 
            tau=0.005, gamma=0.99, actor_update_interval=1, max_size=1e6, layer1_dim=256, layer2_dim=256, 
            batch_size=256, reparam_noise=1e-6, reward_scale=1, loss_type ='MSE', cauchy_scale=0.420, algo_name='SAC'):
        """
        Intialise actor-critic networks and experience replay buffer.

        Parameters:
            env_id (string): name of gym environment
            env (gym object): gym environment
            lr_alpha (float): actor learning rate of Adam optimiser
            lr_beta (float): critic learning rate of Adam optimiser
            lr_kappa (float): temperature learning rate of Adam optimiser
            tau (float<=1): Polyak averaging for target network parameter updates
            gamma (float<=1): discount rate
            actor_update_interval (int): actor policy network update frequnecy
            max_size (int): maximum size of replay buffer
            layer1_dim (int): size of first fully connected layer
            layer2_dim (int): size of second fully connected layer
            batch_size (int): mini-batch size
            reparam_noise (float>0): miniscule constant for valid logarithm
            reward_scale (float): constant factor scaling (inverse temperature) 
            loss_type (str): Cauchy, CE, HSC, Huber, MAE, MSE, TCauchy loss functions
            cauchy_scale (float>0): intialisation value for Cauchy scale parameter
            algo_name (str): name of algorithm
        """
        self.input_dims = input_dims
        self.num_actions = int(num_actions)
        self.max_action = float(env.action_space.high[0])
        self.min_action = float(env.action_space.low[0])
        self.lr_alpha = lr_alpha
        self.lr_beta = lr_beta
        self.lr_kappa = lr_kappa
        self.tau = tau
        self.gamma = gamma

        self.actor_update_interval = int(actor_update_interval)
        self.learn_step_cntr = 0

        self.memory = ReplayBuffer(max_size, self.input_dims, self.num_actions)
        self.batch_size = int(batch_size)
        self.loss_type = str(loss_type)
        self.cauchy_scale = cauchy_scale

        self.reparam_noise = reparam_noise
        self.reward_scale = reward_scale
        self.log_alpha = T.zeros((1,), requires_grad=True)
        self.temp_optimiser = T.optim.Adam([self.log_alpha], lr=self.lr_kappa)
        self.entropy_target = -self.num_actions
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.actor = ActorNetwork(env_id, input_dims, layer1_dim, layer2_dim, num_actions, self.max_action, 
                            self.reparam_noise,lr_alpha, algo_name, loss_type, nn_name='actor')

        self.critic_1 = CriticNetwork(env_id, input_dims, layer1_dim, layer2_dim, num_actions, 
                            self.max_action, lr_beta, algo_name, loss_type, nn_name='critic_1')

        self.critic_2 = CriticNetwork(env_id, input_dims, layer1_dim, layer2_dim, num_actions, 
                                self.max_action, lr_beta, algo_name, loss_type, nn_name='critic_2')

        self.critic_1_target = CriticNetwork(env_id, input_dims, layer1_dim, layer2_dim, num_actions, 
                                self.max_action, lr_beta, algo_name, loss_type, nn_name='critic_1_target')

        self.critic_2_target = CriticNetwork(env_id, input_dims, layer1_dim, layer2_dim, num_actions, 
                                self.max_action, lr_beta, algo_name, loss_type, nn_name='critic_2_target')

        self.update_network_parameters(self.tau)

    def select_next_action(self, state):
        """
        Agent selects next action with sampled form stochastic policy.

        Paramters:
            state (list): current environment state 

        Return:
            next_action: action to be taken by agent in next step
        """        
        # make single state a list for stochastic sampling then select state action
        current_state = T.tensor([state], dtype=T.float).to(self.actor.device)
        action, _ = self.actor.stochastic_mv_gaussian(current_state)
        next_action = action.detach().cpu().numpy()[0]
        # print(next_action)
        return next_action

    def store_transistion(self, state, action, reward, next_state, done):
        """
        Store a transistion to the buffer containing a total up to max_size.

        Paramters:
            state (list): current environment state
            action (list): continuous actions taken to arrive at current state
            reward (float): reward from current environment state
            next_state(list): next environment state
            done (boolean): flag if current state is terminal
        """
        self.memory.store_exp(state, action, reward, next_state, done)

    def learn(self, loss_type):
        """
        Agent learning via SAC algorithm.

        Paramters:
            loss_type (str): Cauchy, CE, Huber, KL, MAE, MSE, TCauchy loss functions

        Returns:
            q1_loss: loss of critic 1
            q2_loss: loss of critic 2
            actor_loss: loss of actor
            scale_1: effective Cauchy scale parameter for critic 1 from Nagy algorithm
            scale_2: effective Cauchy scale parameter for critic 2 from Nagy algorithm
            alpha: entropy adjustment factor (temperature)###################### add to main
        """
        # return nothing till batch size less than replay buffer
        if self.memory.mem_idx < self.batch_size:
            return np.nan, np.nan, np.nan, np.nan, np.nan, self.log_alpha.detach().cpu().numpy()

        states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)

        batch_states = T.tensor(states, dtype=T.float).to(self.critic_1.device)
        batch_actions = T.tensor(actions, dtype=T.float).to(self.critic_1.device)
        batch_rewards = T.tensor(rewards, dtype=T.float).to(self.critic_1.device)
        batch_next_states = T.tensor(next_states, dtype=T.float).to(self.critic_1.device)
        batch_dones = T.tensor(dones, dtype=T.bool).to(self.critic_1.device)
        self.log_alpha = self.log_alpha.to(self.critic_1.device)

        # sample stochastic action policy
        batch_next_stoc_actions, batch_next_logprob_actions = \
                                        self.actor.stochastic_mv_gaussian(batch_next_states)

        # obtain twin soft target Q-values for mini-batch and check terminal status
        q1_target = self.critic_1_target.forward(batch_next_states, batch_next_stoc_actions)
        q2_target = self.critic_2_target.forward(batch_next_states, batch_next_stoc_actions)
        q1_target[batch_dones], q2_target[batch_dones] = 0.0, 0.0
        q1_target, q2_target = q1_target.view(-1), q2_target.view(-1)

        # twin duelling soft target critic values
        soft_q_target = T.min(q1_target, q2_target)
        soft_value = soft_q_target - self.log_alpha.exp() * batch_next_logprob_actions
        target = self.reward_scale * batch_rewards + self.gamma * soft_value
        target = target.view(self.batch_size, -1)

        # obtain twin Q-values for current step
        q1 = self.critic_1.forward(batch_states, batch_actions)
        q2 = self.critic_2.forward(batch_states, batch_actions)

        if self.loss_type == 'TCauchy':
            q1, target = utils.truncation(q1, target)
            q2, target = utils.truncation(q2, target)
        
        # backpropogation of critic loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = utils.loss_function(q1, target, self.loss_type, self.cauchy_scale)
        q2_loss = utils.loss_function(q2, target, self.loss_type, self.cauchy_scale)
        critic_loss = 0.5 * (q1_loss + q2_loss)
        critic_loss.backward(retain_graph=True)

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        # updates Cauchy scale parameter using the Nagy algorithm
        if self.loss_type == 'Cauchy' or 'TCauchy':
            scale_1 = utils.nagy_algo(q1, target, self.cauchy_scale)
            scale_2 = utils.nagy_algo(q2, target, self.cauchy_scale)
            self.cauchy_scale = (scale_1 + scale_2)/2

        # update actor, temperature and target critic networks every interval
        if self.learn_step_cntr % self.actor_update_interval != 0:
            numpy_q1_loss = q1_loss.detach().cpu().numpy()
            numpy_q2_loss = q2_loss.detach().cpu().numpy()
            return numpy_q1_loss, numpy_q2_loss, np.nan, scale_1, scale_2, np.nan

        q1 = self.critic_1.forward(batch_states, batch_actions)
        q2 = self.critic_2.forward(batch_states, batch_actions)
        soft_q = T.min(q1, q2).view(-1)

        self.actor.optimizer.zero_grad()

        actor_loss = self.log_alpha.exp() * batch_next_logprob_actions - soft_q.clone()
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        
        self.actor.optimizer.step()

        self.temp_optimiser.zero_grad()

        temp_loss = -self.log_alpha.exp() * (batch_next_logprob_actions.detach() + self.entropy_target)
        temp_loss = temp_loss.mean()
        temp_loss.backward()

        self.temp_optimiser.step()

        self.update_network_parameters(self.tau)

        numpy_q1_loss = q1_loss.detach().cpu().numpy()
        numpy_q2_loss = q2_loss.detach().cpu().numpy()
        numpy_actor_loss = actor_loss.detach().cpu().numpy()
        numpy_logtemp = self.log_alpha.detach().cpu().numpy()
        return numpy_q1_loss, numpy_q2_loss, numpy_actor_loss, scale_1, scale_2, numpy_logtemp


    def update_network_parameters(self, tau):
        """
        Update target network parameters with smoothing.

        Paramters:
            tau (float<=1): Polyak averaging rate for target network parameter updates
        """        
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self):
        """
        Saves all 6 networks.
        """
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.critic_1_target.save_checkpoint()
        self.critic_2_target.save_checkpoint()

    def load_models(self):
        """
        Loads all 6 networks.
        """
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.critic_1_target.load_checkpoint()
        self.critic_2_target.load_checkpoint()