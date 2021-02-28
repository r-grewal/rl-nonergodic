import numpy as np
import torch as T
from replay import ReplayBuffer
from networks_td3 import ActorNetwork, CriticNetwork
import utils 

class Agent_td3():
    """
    TD3 agent algorithm.
    """
    def __init__(self, env_id, env, input_dims, num_actions, lr_alpha=0.001, lr_beta=0.001, 
            tau=0.005, gamma=0.99, actor_update_interval=2, warmup=1000, max_size=1e6, 
            layer1_dim=400, layer2_dim=300, batch_size=100, policy_noise=0.1, target_policy_noise=0.2, 
            target_policy_clip=0.5, loss_type ='MSE', cauchy_scale=0.420, algo_name='TD3'):
        """
        Intialise actor-critic networks and experience replay buffer.

        Parameters:
            env_id (string): name of gym environment
            env (gym object): gym environment
            lr_alpha (float): actor learning rate of Adam optimiser
            lr_beta (float): critic learning rate of Adam optimiser
            tau (float<=1): Polyak averaging for target network parameter updates
            gamma (float<=1): discount rate
            actor_update_interval (int): actor policy network update frequnecy
            warmup (int): warmup interval of random actions steps per episode
            max_size (int): maximum size of replay buffer
            layer1_dim (int): size of first fully connected layer
            layer2_dim (int): size of second fully connected layer
            batch_size (int): mini-batch size
            noise (float): action exploration Gaussian noise (standard deviation)
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
        self.tau = tau
        self.gamma = gamma

        self.actor_update_interval = int(actor_update_interval)
        self.warmup = int(warmup)
        self.learn_step_cntr = 0
        self.time_step = 0

        self.memory = ReplayBuffer(max_size, self.input_dims, self.num_actions)
        self.batch_size = int(batch_size)
        self.policy_noise = policy_noise
        self.target_policy_noise = target_policy_noise
        self.target_policy_clip = target_policy_clip

        self.loss_type = str(loss_type)
        self.cauchy_scale = cauchy_scale

        self.actor = ActorNetwork(env_id, input_dims, layer1_dim, layer2_dim, num_actions, 
                            self.max_action, lr_alpha, algo_name, loss_type, nn_name='actor')
        
        self.actor_target = ActorNetwork(env_id, input_dims, layer1_dim, layer2_dim, num_actions, 
                            self.max_action, lr_alpha, algo_name, loss_type, nn_name='actor_target')

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
        Agent selects next action with added noise to each component or during warmup a random action taken.

        Parameters:
            state (list): current environment state 

        Return:
            next_action: action to be taken by agent in next step
        """
        # action = env.action_space.sample()
        action_noise = T.tensor(np.random.normal(loc=0, scale=self.policy_noise, 
                            size=(self.num_actions,)), dtype=T.float)
        mu = action_noise.to(self.actor.device)

        if self.time_step > self.warmup:
            current_state = T.tensor(state, dtype=T.float).to(self.actor.device)
            mu += self.actor.forward(current_state)

        action = T.clamp(mu, self.min_action, self.max_action)
        print(action)
        next_action = action.detach().cpu().numpy()

        self.time_step += 1
        
        return next_action

    def store_transistion(self, state, action, reward, next_state, done):
        """
        Store a transistion to the buffer containing a total up to max_size.

        Parameters:
            state (list): current environment state
            action (list): continuous actions taken to arrive at current state
            reward (float): reward from current environment state
            next_state(list): next environment state
            done (boolean): flag if current state is terminal
        """
        self.memory.store_exp(state, action, reward, next_state, done)

    def learn(self, loss_type):
        """
        Agent learning via TD3 algorithm.

        Parameters:
            loss_type (str): Cauchy, CE, Huber, KL, MAE, MSE, TCauchy loss functions

        Returns:
            q1_loss: loss of critic 1
            q2_loss: loss of critic 2
            actor_loss: loss of actor
            scale_1: effective Cauchy scale parameter for critic 1 from Nagy algorithm
            scale_2: effective Cauchy scale parameter for critic 2 from Nagy algorithm
        """
        # return nothing till batch size less than replay buffer
        if self.memory.mem_idx < self.batch_size:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)

        batch_states = T.tensor(states, dtype=T.float).to(self.critic_1.device)
        batch_actions = T.tensor(actions, dtype=T.float).to(self.critic_1.device)
        batch_rewards = T.tensor(rewards, dtype=T.float).to(self.critic_1.device)
        batch_next_states = T.tensor(next_states, dtype=T.float).to(self.critic_1.device)
        batch_dones = T.tensor(dones, dtype=T.bool).to(self.critic_1.device) 

        # add random noise to Gaussian noise to each component of next action with clipping
        target_action_noise = T.tensor(np.random.normal(loc=0, scale=self.target_policy_noise, 
                size=(batch_actions.shape)), dtype=T.float).clamp(-self.target_policy_clip, self.target_policy_clip)
        target_action_noise = target_action_noise.to(self.actor.device)
        
        batch_next_actions = self.actor_target.forward(batch_next_states)
        batch_next_actions = (batch_next_actions + target_action_noise).clamp(self.min_action, self.max_action)

        # obtain twin target Q-values for mini-batch and check terminal status
        q1_target = self.critic_1_target.forward(batch_next_states, batch_next_actions)
        q2_target = self.critic_2_target.forward(batch_next_states, batch_next_actions)
        q1_target[batch_dones], q2_target[batch_dones] = 0.0, 0.0
        q1_target, q2_target = q1_target.view(-1), q2_target.view(-1)
        
        # twin duelling target critic values
        q_target = T.min(q1_target, q2_target)
        target = batch_rewards + self.gamma * q_target
        target = target.view(self.batch_size, 1)

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

        critic_loss = q1_loss + q2_loss 
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        # updates Cauchy scale parameter using the Nagy algorithm
        scale_1 = utils.nagy_algo(q1, target, self.cauchy_scale)
        scale_2 = utils.nagy_algo(q2, target, self.cauchy_scale)
        self.cauchy_scale = (scale_1 + scale_2)/2

        # update actor and all target networks every interval
        if self.learn_step_cntr % self.actor_update_interval != 0:
            numpy_q1_loss = q1_loss.detach().cpu().numpy()
            numpy_q2_loss = q2_loss.detach().cpu().numpy()
            return numpy_q1_loss, numpy_q2_loss, np.nan, scale_1, scale_2, np.nan

        self.actor.optimizer.zero_grad()

        actor_q1_loss = self.critic_1.forward(batch_states, self.actor.forward(batch_states))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        
        self.actor.optimizer.step()

        self.update_network_parameters(self.tau)

        numpy_q1_loss = q1_loss.detach().cpu().numpy()
        numpy_q2_loss = q2_loss.detach().cpu().numpy()
        numpy_actor_loss = actor_loss.detach().cpu().numpy()    
        return numpy_q1_loss, numpy_q2_loss, numpy_actor_loss, scale_1, scale_2, np.nan

    def update_network_parameters(self, tau):
        """
        Update target network parameters with smoothing.

        Parameters:
            tau (float<=1): Polyak averaging rate for target network parameter updates
        """
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self):
        """
        Saves all 6 networks.
        """
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.critic_1_target.save_checkpoint()
        self.critic_2_target.save_checkpoint()

    def load_models(self):
        """
        Loads all 6 networks.
        """
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.critic_1_target.load_checkpoint()
        self.critic_2_target.load_checkpoint()