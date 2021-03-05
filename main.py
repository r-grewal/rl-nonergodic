import os
import time
import gym
import pybullet_envs
import numpy as np
from algo_td3 import Agent_td3
from algo_sac import Agent_sac
from utils import plot_learning_curve

algo = 'TD3'            # set to 'TD3' or 'SAC'
loss_fn = 'MSE'         # set to 'Cauchy', 'HSC', 'Huber', 'MAE', 'MSE' or 'TCauchy'
multi_steps = 1        # set bootstrapping of target values
ergodicity = 'yes'      # whether to assume ergodicity
buffer = 1e6            # size of experience replay buffer
discount = 0.99         # discount rate per step
random_steps = 1000     # intial random steps for TD3
scale = 1               # reward scale for SAC
sdist = 'mvn'           # set stochastic sampling 'beta', 'mvn', 'st' or 'uvn' for SAC

n_trials = 1            # number of total trials
n_episodes = 500      # maximum number of episodes per trial
n_cumsteps = 3e6        # maximum cumulative steps per trial regardless of episodes

openai = ['LunarLanderContinuous-v2', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3']
pybullet = ['CartPoleContinuousBulletEnv-v0', 'InvertedPendulumBulletEnv-v0','InvertedDoublePendulumBulletEnv-v0',      
            'KukaBulletEnv-v0', 'HopperBulletEnv-v0', 'Walker2DBulletEnv-v0',
            'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0', 'HumanoidBulletEnv-v0']

if __name__ == '__main__':

    env_id = pybullet[2]
    env = gym.make(env_id)
    env = env.unwrapped
    # env._max_episode_steps = 5000
    input_dims = env.observation_space.shape
    num_actions = env.action_space.shape[0]

    if algo == 'TD3':
        batch_size = 100
        agent = Agent_td3(env_id, env, input_dims, num_actions, lr_alpha=0.001, lr_beta=0.001, tau=0.005, 
                gamma=0.99, actor_update_interval=2, warmup=random_steps, max_size=buffer, layer1_dim=400, 
                layer2_dim=300, batch_size=batch_size, policy_noise=0.1, target_policy_noise=0.2, 
                target_policy_clip=0.5,loss_type=loss_fn, cauchy_scale=0.420, algo_name=algo, erg=ergodicity)

    elif algo == 'SAC':
        batch_size = 256
        agent = Agent_sac(env_id, env, input_dims, num_actions, lr_alpha=3e-4, lr_beta=3e-4, lr_kappa=3e-4, 
                tau=0.005, gamma=0.99, actor_update_interval=1, max_size=buffer, layer1_dim=256, layer2_dim=256, 
                batch_size=batch_size, reparam_noise=1e-6, reward_scale=scale, loss_type =loss_fn, 
                cauchy_scale=0.420, stoch=sdist, algo_name=algo, erg=ergodicity)

    for round in range(n_trials):
        best_score = env.reward_range[0] # set intial best to worst possible reward
        score_history = []
        trial_log = np.zeros((n_episodes+1, 9))
        cum_steps = 0
        epis = 0

        # agent.load_models()

        while epis <= (n_episodes): # or cum_steps <= n_cumsteps:
            start_time = time.perf_counter()

            state = env.reset()
            done = False

            score = 0
            step = 0

            while not done:
                # action = env.action_space.sample()
                action = agent.select_next_action(state, multi='no')
                next_state, reward, done, info = env.step(action)

                batch_states, batch_actions, batch_rewards, \
                                    batch_next_states, batch_dones = agent.get_mini_batch(batch_size)

                # print(batch_states)
                bat_targets = agent.multi_step_target(batch_actions, batch_rewards, batch_next_states, \
                                                      batch_dones, env, multi_steps, ergodicity)
                # print(bat_targets)
                q1_loss, q2_loss, actor_loss, scale_1, scale_2, logtemp = \
                            agent.learn(batch_states, batch_actions, bat_targets, loss_fn, ergodicity)

                agent.store_transistion(state, action, reward, next_state, done)

                # env.reset()                                                               
                env.state = next_state

                state = next_state
                score += reward
                step += 1
                cum_steps += 1
                # print(step)

                # env.render(mode='human')

            # env.close()
            end_time = time.perf_counter()

            score_history.append(score)
            trail_score = np.mean(score_history[-100:])
            trial_log[epis, 0], trial_log[epis, 2], trial_log[epis, 8] = end_time - start_time, step, logtemp
            trial_log[epis, 3:6] = np.array([q1_loss, q2_loss, actor_loss])
            trial_log[epis, 6:8] = np.array([scale_1, scale_2])
             
            if trail_score > best_score:
                best_score = trail_score
                agent.save_models()

            print('eps/step/tot {}/{}/{}: score {:1.1f}, trail {:1.0f}, C/A loss {:1.1f}/{:1.1f}, min {:1.1f}'
                .format(epis+1, step, cum_steps, score, trail_score, \
                        q1_loss+q2_loss, actor_loss, (end_time-start_time)/60))
            
            epis += 1

        trial_log[:, 1] = np.array(score_history)

        directory = 'results/'+env_id+'/'+env_id+'_'+algo+'_'+loss_fn+'_'+'e'+str(n_episodes)+'_'+'n'
        filename_npy = directory + str(round+1)+'_log'+'.npy'
        filename_png = directory + str(round+1)+'.png'

        if not os.path.exists('./results/'+env_id):
            os.makedirs('./results/'+env_id)

        np.save(filename_npy, trial_log)

        plot_learning_curve(score_history[1:], filename_png)
