import gym
import pybullet_envs
import numpy as np
import torch as T
from replay import ReplayBuffer
from networks_sac import ActorNetwork, CriticNetwork
import utils 

class Agent_sac():
    """
    SAC agent algorithm based on https://arxiv.org/pdf/1812.05905.pdf.
    """                             
    def __init__(self, env_id, env, lr_alpha=3e-4, lr_beta=3e-4, lr_kappa=3e-4, tau=0.005, layer1_dim=256, 
                 layer2_dim=256, cauchy_scale_1=0.420, cauchy_scale_2=0.420, warmup=1000, gamma=0.99, erg='Yes', 
                 loss_type='MSE', max_size=1e6, algo_name='SAC', actor_update_interval=1, batch_size=256, 
                 reparam_noise=1e-8, reward_scale=1, stoch='N'):
        """
        Intialise actor-critic networks and experience replay buffer.

        Parameters:
            env_id (string): name of gym environment
            env (gym object): gym environment
            lr_alpha (float>0): actor learning rate of Adam optimiser
            lr_beta (float>0): critic learning rate of Adam optimiser
            lr_kappa (float>0): temperature learning rate of Adam optimiser
            tau (float<=1): Polyak averaging for target network parameter updates
            layer1_dim (int): size of first fully connected layer
            layer2_dim (int): size of second fully connected layer
            cauchy_scale (float>0): intialisation value for Cauchy scale parameter
            warmup (int): intial random warmup steps to generate random seed
            gamma (float<=1): discount factor
            erg (str): whether to assume ergodicity
            loss_type (str): critic loss functions
            max_size (int): maximum size of replay buffer
            algo_name (str): name of algorithm
            actor_update_interval (int): actor policy network update frequnecy
            batch_size (int): mini-batch size
            reparam_noise (float>0): miniscule constant to keep logarithm bounded
            reward_scale (float): constant factor scaling ('inverse temperature')
            stoch (str): stochastic actor policy sampling distribution
        """
        self.env_id = env_id
        self.env = gym.make(self.env_id)
        self.env = self.env.unwrapped
        state = self.env.reset()

        self.input_dims = env.observation_space.shape    # input dimensions tuple
        self.num_actions = int(env.action_space.shape[0])
        self.max_action = float(env.action_space.high[0])
        self.min_action = float(env.action_space.low[0])

        self.lr_alpha = lr_alpha
        self.lr_beta = lr_beta
        self.lr_kappa = lr_kappa
        self.tau = tau
        self.cauchy_scale_1 = cauchy_scale_1
        self.cauchy_scale_2 = cauchy_scale_2
        self.warmup = int(warmup)
        self.gamma = gamma
        self.erg = str(erg)
        self.loss_type = str(loss_type)

        self.memory = ReplayBuffer(max_size, self.input_dims, self.num_actions)
        self.actor_update_interval = int(actor_update_interval)
        self.batch_size = int(batch_size)

        self.reparam_noise = reparam_noise
        self.reward_scale = reward_scale
        self.stoch = str(stoch)

        self.time_step = 0
        self.learn_step_cntr = 0

        self.actor = ActorNetwork(env_id, self.input_dims, layer1_dim, layer2_dim, self.num_actions, 
                            self.max_action, self.reparam_noise, lr_alpha, algo_name, loss_type, nn_name='actor')

        self.critic_1 = CriticNetwork(env_id, self.input_dims, layer1_dim, layer2_dim, self.num_actions, 
                            self.max_action, lr_beta, algo_name, loss_type, nn_name='critic_1')

        self.critic_2 = CriticNetwork(env_id, self.input_dims, layer1_dim, layer2_dim, self.num_actions, 
                                self.max_action, lr_beta, algo_name, loss_type, nn_name='critic_2')

        self.critic_1_target = CriticNetwork(env_id, self.input_dims, layer1_dim, layer2_dim, self.num_actions, 
                                self.max_action, lr_beta, algo_name, loss_type, nn_name='critic_1_target')

        self.critic_2_target = CriticNetwork(env_id, self.input_dims, layer1_dim, layer2_dim, self.num_actions, 
                                self.max_action, lr_beta, algo_name, loss_type, nn_name='critic_2_target')

        # learn temperature via convex optimisation (gradient descent)
        self.entropy_target = -self.num_actions    # heuristic assumption
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.log_alpha = T.zeros((1,), requires_grad=True, device=self.device)
        self.temp_optimiser = T.optim.Adam([self.log_alpha], lr=self.lr_kappa)
        
        self.update_network_parameters(self.tau)

        batch_next_states = T.zeros((self.batch_size, sum(self.input_dims)), requires_grad=True).to(self.actor.device)
        batch_rewards = T.zeros((self.batch_size, ), requires_grad=True).to(self.actor.device)
        # self.select_next_action(batch_next_states, self.stoch, multi='No')
        self.single_step_target(batch_rewards, batch_next_states, None, self.stoch, self.erg)

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

    def get_mini_batch(self, batch_size):
        """
        Uniform sampling from replay buffer and send to GPU.

        Parameters:
            batch_size (int): mini-batch size

        Returns:
            states (array): batch of environment states
            actions (array): batch of continuous actions taken to arrive at states
            rewards (array): batch of rewards from current states
            next_states (array): batch of next environment states
            dones (array): batch of done flags
        """
        if self.memory.mem_idx < self.batch_size + 1:
            return np.nan, np.nan, np.nan, np.nan, np.nan
            
        states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)

        batch_states = T.tensor(states, dtype=T.float).to(self.critic_1.device)
        batch_actions = T.tensor(actions, dtype=T.float).to(self.critic_1.device)
        batch_rewards = T.tensor(rewards, dtype=T.float).to(self.critic_1.device)
        batch_next_states = T.tensor(next_states, dtype=T.float).to(self.critic_1.device)
        batch_dones = T.tensor(dones, dtype=T.bool).to(self.critic_1.device) 

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def select_next_action(self, state, stoch='N', multi='No'):
        """
        Agent selects next action with added noise to each component or during warmup a random action taken.

        Parameters:
            state (list): current environment state
            stoch (str): stochastic policy sampling via 'L' 'MVN', 'T' or 'N' distribution
            multi (str): whether action is being taken as part of n-step targets

        Return:
            numpy_next_action: action to be taken by agent in next step for gym
            next_action: action to be taken by agent in next step
        """        
        if self.time_step > self.warmup:
            # make single state a list for stochastic sampling then select state action
            current_state = T.tensor([state], dtype=T.float).to(self.actor.device)

            if self.stoch != 'MVN':
                next_action, _ = self.actor.stochastic_uv(current_state, self.stoch)
            else:
                next_action, _ = self.actor.stochastic_mv_gaussian(current_state)
            
            numpy_next_action = next_action.detach().cpu().numpy()[0]
        
        else:
            numpy_next_action = self.env.action_space.sample()
            next_action = T.tensor(numpy_next_action, dtype=T.float).to(self.actor.device)

        if multi == 'No':
            self.time_step += 1

        return numpy_next_action, next_action
    
    def single_step_target(self, batch_rewards, batch_next_states, batch_dones, stoch='N', erg='Yes'):
        """
        Standard single step target Q-values for mini-batch.

        Parameters:
            batch_rewards (array): batch of rewards from current states
            batch_next_states (array): batch of next environment states
            batch_dones (array): batch of done flags
            stoch (str): stochastic policy sampling via 'L' 'MVN', 'T' or 'N' distribution
            erg (str): whether to assume ergodicity
        
        Returns:
            batch_target (array): twin duelling target Q-values
        """
        # sample next stochastic action policy for target critic network based on mini-batch
        if self.stoch != 'MVN':
            batch_next_stoc_actions, batch_next_logprob_actions = \
                            self.actor.stochastic_uv(batch_next_states, self.stoch)
        else:
            batch_next_stoc_actions, batch_next_logprob_actions = \
                                        self.actor.stochastic_mv_gaussian(batch_next_states)

        batch_next_logprob_actions = batch_next_logprob_actions.view(-1)

        # obtain twin next soft target Q-values for mini-batch and check terminal status
        q1_target = self.critic_1_target.forward(batch_next_states, batch_next_stoc_actions)
        q2_target = self.critic_2_target.forward(batch_next_states, batch_next_stoc_actions)
        q1_target[batch_dones], q2_target[batch_dones] = 0.0, 0.0
        q1_target, q2_target = q1_target.view(-1), q2_target.view(-1)
    
        # twin duelling soft target critic values
        soft_q_target = T.min(q1_target, q2_target)
        soft_value = soft_q_target - self.log_alpha.exp() * batch_next_logprob_actions
        target = self.reward_scale * batch_rewards + self.gamma * soft_value
        batch_target = target.view(self.batch_size, -1)

        return batch_target

    def multi_step_target(self, batch_rewards, batch_next_states, batch_dones, env, stoch='N', erg='Yes', n_step=1):
        """
        Multi-step target Q-values for mini-batch based on repeatedly propogating n times through policy network 
        using greedy action selection with added noise to simulate additional steps from the current environment state
        and the targets then become: \Sum(\gamma^{k} * R_{k+1}, k=0, n-1) + \gamma^{n}*min(Q_1, Q2). 

        Parameters:
            batch_rewards (array): batch of rewards from current states
            batch_next_states (array): batch of next environment states
            batch_dones (array): batch of done flags
            env (gym object): gym environment
            stoch (str): stochastic policy 'L' 'MVN', 'T' or 'N' distribution
            erg (str): whether to assume ergodicity
            n_steps (int): number of steps of greedy action selection to take
        
        Returns:
            batch_target (array): twin duelling multi-step target Q-values
        """
        if self.memory.mem_idx < self.batch_size + 10:    # add +10 offset to account for ...
            return np.nan
        
        n_step = int(n_step)

        if n_step <= 1:
            batch_target = self.single_step_target(batch_rewards, batch_next_states, batch_dones, self.stoch, self.erg)
            return batch_target
    
        print('MULTI-STEP NOT YET IMPLEMENTED')

    def learn(self, batch_states, batch_actions, batch_target, loss_type, stoch='N', erg='Yes'):
        """
        Agent learning via SAC algorithm.

        Parameters:
            batch_next_states (array): batch of current environment states
            batch_actions (array): batch of continuous actions taken to arrive at states
            batch_target (array): twin duelling target Q-values
            loss_type (str): surrogate loss functions
            stoch (str): stochastic policy 'L' 'MVN', 'T' or 'N' distribution
            erg (str): whether to assume ergodicity

        Returns:
            q1_loss: loss of critic 1
            q2_loss: loss of critic 2
            actor_loss: loss of actor
            scale_1: effective Cauchy scale parameter for critic 1 from Nagy algorithm
            scale_2: effective Cauchy scale parameter for critic 2 from Nagy algorithm
            logtemp: log entropy adjustment factor (temperature)
        """
        # return nothing till batch size less than replay buffer
        if self.memory.mem_idx < self.batch_size + 10:
            loss = [np.nan, np.nan, np.nan]
            cpu_logtmep = self.log_alpha.detach().cpu().numpy()[0]
            loss_params = [np.nan, np.nan, np.nan, np.nan]
            return loss, cpu_logtmep, loss_params

        # obtain current twin soft Q-values for mini-batch
        q1 = self.critic_1.forward(batch_states, batch_actions)
        q2 = self.critic_2.forward(batch_states, batch_actions)
        
        # updates CIM size empircally
        kernel_1 = utils.cim_size(q1, batch_target)
        kernel_2 = utils.cim_size(q2, batch_target)

        # backpropogation of critic loss while retaining graph due to coupling
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = utils.loss_function(q1, batch_target, self.loss_type, self.cauchy_scale_1, kernel_1)
        q2_loss = utils.loss_function(q2, batch_target, self.loss_type, self.cauchy_scale_2, kernel_2)
    
        critic_loss = 0.5 * (q1_loss + q2_loss)
        critic_loss.backward(retain_graph=True)

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # updates Cauchy scale parameter using the Nagy algorithm
        self.cauchy_scale_1 = utils.nagy_algo(q1, batch_target, self.cauchy_scale_1)
        self.cauchy_scale_2 = utils.nagy_algo(q2, batch_target, self.cauchy_scale_2)

        self.learn_step_cntr += 1

        cpu_q1_loss = q1_loss.detach().cpu().numpy()
        cpu_q2_loss = q2_loss.detach().cpu().numpy()
        cpu_logtmep = self.log_alpha.detach().cpu().numpy()[0]

        loss = [cpu_q1_loss , cpu_q2_loss, np.nan] 
        loss_params = [self.cauchy_scale_1, self.cauchy_scale_2, kernel_1, kernel_2]

        # update actor, temperature and target critic networks every interval
        if self.learn_step_cntr % self.actor_update_interval != 0:
            return loss, cpu_logtmep, loss_params

        # sample current stochastic action policy for critic network based on mini-batch
        if self.stoch != 'MVN':
            batch_stoc_actions, batch_logprob_actions = \
                                        self.actor.stochastic_uv(batch_states, self.stoch)
        else:
            batch_stoc_actions, batch_logprob_actions = \
                                        self.actor.stochastic_mv_gaussian(batch_states)

        batch_logprob_actions = batch_logprob_actions.view(-1)

        # obtain twin current soft-Q values for mini-batch using stochastic sampling
        q1 = self.critic_1.forward(batch_states, batch_stoc_actions)
        q2 = self.critic_2.forward(batch_states, batch_stoc_actions)
        soft_q = T.min(q1, q2).view(-1)

        # learn stochastic actor policy
        self.actor.optimizer.zero_grad()

        actor_loss = self.log_alpha.exp() * batch_logprob_actions - soft_q
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        
        self.actor.optimizer.step()

        # learn temperature by approximating gradient
        self.temp_optimiser.zero_grad()

        temp_loss = -self.log_alpha.exp() * (batch_logprob_actions.detach() + self.entropy_target)
        temp_loss = temp_loss.mean()
        temp_loss.backward()

        self.temp_optimiser.step()

        self.update_network_parameters(self.tau)

        cpu_actor_loss = actor_loss.detach().cpu().numpy()
        loss[2] = cpu_actor_loss
        cpu_logtmep = self.log_alpha.detach().cpu().numpy()[0]
        
        return loss, cpu_logtmep, loss_params

    def update_network_parameters(self, tau):
        """
        Update target network parameters with smoothing.

        Parameters:
            tau (float<=1): Polyak averaging rate for target network parameter updates
        """        
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self):
        """
        Saves all 3 networks.
        """
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        """
        Loads all 3 networks.
        """
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
