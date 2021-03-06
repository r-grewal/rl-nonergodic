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
            target_policy_clip=0.5, loss_type ='MSE', cauchy_scale=0.420, algo_name='TD3', erg='Yes'):
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
            erg (str): whether to assume ergodicity
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
        self.policy_noise = policy_noise * self.max_action
        self.target_policy_noise = target_policy_noise * self.max_action
        self.target_policy_clip = target_policy_clip * self.max_action

        self.loss_type = str(loss_type)
        self.cauchy_scale = cauchy_scale
        self.erg = str(erg)

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

        batch_next_states = T.zeros((self.batch_size, *self.input_dims), requires_grad=True).to(self.actor.device)
        batch_rewards = T.zeros((self.batch_size, ), requires_grad=True).to(self.actor.device)
        self.select_next_action(batch_next_states, multi='no')
        self.single_step_target(batch_rewards, batch_next_states, None, self.erg)
        self.update_network_parameters(self.tau)

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

        Paramters:
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

    def select_next_action(self, state, multi='no'):
        """
        Agent selects next action with added noise to each component or during warmup a random action taken.

        Parameters:
            state (list): current environment state
            multi (str): whether action is being taken as part of n-step targets

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
        next_action = action.detach().cpu().numpy()

        if multi == 'no':
            self.time_step += 1
        
        return next_action

    def single_step_target(self, batch_rewards, batch_next_states, batch_dones, erg='Yes'):
        """
        Standard single step target Q-values for mini-batch.

        Parameters:
            batch_rewards (array): batch of rewards from current states
            batch_next_states (array): batch of next environment states
            batch_dones (array): batch of done flags
            erg (str): whether to assume ergodicity
        
        Returns:
            batch_target (array): twin duelling target Q-values
        """
        # add random Gaussian noise to each component of next target action with clipping
        target_action_noise = T.tensor(np.random.normal(loc=0, scale=self.target_policy_noise, 
                                            size=(self.batch_size, self.num_actions)), dtype=T.float)
        target_action_noise = target_action_noise.clamp(-self.target_policy_clip, self.target_policy_clip)
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
        batch_target = target.view(self.batch_size, 1)

        return batch_target

    def multi_step_target(self, batch_rewards, batch_next_states, batch_dones, env, n_step=1, erg='Yes'):
        """
        Multi-step target Q-values for mini-batch based on repeatedly propogating n times through policy network 
        using greedy action selection with added noise to simulate additional steps from the current environment state
        and the targets then become: \Sum(\gamma^{k} * R_{k+1}, k=0, n-1) + \gamma^{n}*min(Q_1, Q2). 

        Parameters:
            batch_rewards (array): batch of rewards from current states
            batch_next_states (array): batch of next environment states
            batch_dones (array): batch of done flags
            env (gym object): gym environment
            n_steps (int): number of steps of greedy action selection to take
            erg (str): whether to assume ergodicity
        
        Returns:
            batch_target (array): twin duelling multi-step target Q-values
        """
        if self.memory.mem_idx < self.batch_size + 10:    # add +10 offset to account for ...
                return np.nan
        
        n_step = int(n_step)

        if n_step <= 1:
            batch_target = self.single_step_target(batch_rewards, batch_next_states, batch_dones, erg)
            return batch_target

        # create appropriately sized tensors for multi-step actions
        batch_multi_next1_actions = T.zeros((self.batch_size, self.num_actions), requires_grad=True).to(self.actor.device)
        batch_multi_next2_actions = T.zeros((self.batch_size, self.num_actions), requires_grad=True).to(self.actor.device)
        batch_multi_next1_rewards = batch_rewards.to(self.actor.device).clone()
        batch_multi_next1_states  = batch_next_states.to(self.actor.device).clone()
        batch_multi_next2_states  = batch_next_states.to(self.actor.device).clone()
        batch_multi_next1_dones = batch_dones.to(self.actor.device).clone()
        sample = T.zeros((self.batch_size, *self.input_dims), requires_grad=True).to(self.actor.device)

        # print(batch_multi_next1_rewards, batch_multi_next1_rewards.grad)

        # multi-step actions per sample contained in mini-batch
        for steps in range(n_step - 1):
            # print(batch_multi_next1_rewards, batch_multi_next1_rewards.grad)
            for n in range(self.batch_size):
                # while not batch_multi_next_dones[n].detach().cpu().numpy():
                if batch_multi_next1_dones[n].detach().cpu().numpy() == False:
                    # print(batch_multi_next1_states, batch_multi_next1_states.size())
                    sample[n] = batch_multi_next1_states[n]
                    # print(sample)

                    # add random Gaussian noise to each actual component of next action with clipping 
                    current_state = sample[n].detach().cpu().numpy()
                    next_action = self.select_next_action(current_state, multi='yes')
                    # print(next_action)
                    batch_multi_next1_actions[n] = T.tensor(next_action, dtype=T.float).to(self.actor.device)
                    # print(batch_multi_next1_actions)
                    env.state = sample[n].detach().cpu().numpy()
                    # print(env.state)
                    # simulate next environement step
                    next_next_state, multi_reward, multi_done, _ = env.step(next_action)
                    # print(next_next_state)

                    # add next reward and update done flags
                    batch_multi_next1_rewards[n] += multi_reward * self.gamma**(steps + 1)
                    batch_multi_next1_dones[n] = T.tensor(multi_done, dtype=T.bool)
                    batch_multi_next2_states[n] = T.tensor(next_next_state, dtype=T.float).to(self.actor.device)

            # print(batch_multi_next1_rewards, batch_multi_next1_rewards.grad)
            # print(batch_multi_next1_actions, batch_multi_next1_actions.grad)

            # add random Gaussian noise to each target component of next action with clipping
            target_action_noise = T.tensor(np.random.normal(loc=0, scale=self.target_policy_noise, 
                                                size=(self.batch_size, self.num_actions)), dtype=T.float)
            target_action_noise = target_action_noise.clamp(-self.target_policy_clip, self.target_policy_clip)
            target_action_noise = target_action_noise.to(self.actor.device)

            batch_multi_next2_actions = self.actor_target.forward(batch_multi_next2_states)
            batch_multi_next2_actions = (batch_multi_next2_actions + target_action_noise).clamp(self.min_action, self.max_action)

            # print(batch_multi_next2_actions, batch_multi_next2_actions.grad)

            # obtain twin target Q-values for mini-batch and check terminal status
            q1_target = self.critic_1_target.forward(batch_multi_next2_states, batch_multi_next2_actions)
            q2_target = self.critic_2_target.forward(batch_multi_next2_states, batch_multi_next2_actions)
            q1_target[batch_multi_next1_dones], q2_target[batch_multi_next1_dones] = 0.0, 0.0
            q1_target, q2_target = q1_target.view(-1), q2_target.view(-1)

            # twin duelling target critic values                
            batch_multi_q = T.min(q1_target, q2_target)
            batch_multi_target = batch_multi_next1_rewards + self.gamma**(steps + 1) * batch_multi_q

            # print(batch_multi_target, batch_multi_target.grad)
        # print(batch_multi_next1_rewards, batch_multi_next1_rewards.grad)
        batch_target = batch_multi_target.view(self.batch_size, 1)
        # print(batch_target, batch_target.grad)

        return batch_target

    def learn(self, batch_states, batch_actions, batch_target, loss_type, erg='Yes'):
        """
        Agent learning via TD3 algorithm with multi-step bootstrapping.

        Parameters:
            batch_next_states (array): batch of current environment states
            batch_actions (array): batch of continuous actions taken to arrive at states
            batch_target (array): twin duelling target Q-values
            loss_type (str): Cauchy, Huber, MAE, MSE, TCauchy loss functions

        Returns:
            q1_loss: loss of critic 1
            q2_loss: loss of critic 2
            actor_loss: loss of actor
            scale_1: effective Cauchy scale parameter for critic 1 from Nagy algorithm
            scale_2: effective Cauchy scale parameter for critic 2 from Nagy algorithm
            logtemp: N/A
        """
        # return nothing till batch size less than replay buffer
        if self.memory.mem_idx < self.batch_size + 10:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # obtain twin Q-values for current step
        q1 = self.critic_1.forward(batch_states, batch_actions)
        q2 = self.critic_2.forward(batch_states, batch_actions)
        
        # backpropogation of critic loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = utils.loss_function(q1, batch_target, self.loss_type, self.cauchy_scale)
        q2_loss = utils.loss_function(q2, batch_target, self.loss_type, self.cauchy_scale)
        
        critic_loss = q1_loss + q2_loss 

        # print(q1_loss, q2_loss, critic_loss)
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        # updates Cauchy scale parameter using the Nagy algorithm
        scale_1 = utils.nagy_algo(q1, batch_target, self.cauchy_scale)
        scale_2 = utils.nagy_algo(q2, batch_target, self.cauchy_scale)
        self.cauchy_scale = (scale_1 + scale_2)/2

        # update actor and all target networks every interval
        if self.learn_step_cntr % self.actor_update_interval != 0:
            numpy_q1_loss = q1_loss.detach().cpu().numpy()
            numpy_q2_loss = q2_loss.detach().cpu().numpy()
            return numpy_q1_loss, numpy_q2_loss, np.nan, scale_1, scale_2, np.nan

        # DDPG gradient ascent via backpropogation i.e. function approximation
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
        Update target deep network parameters with smoothing.

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
