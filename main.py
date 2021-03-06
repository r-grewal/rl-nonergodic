import os
import time
import gym
import pybullet_envs
import numpy as np
from algo_td3 import Agent_td3
from algo_sac import Agent_sac
from utils import plot_learning_curve

gym_envs = [#'LunarLanderContinuous-v2', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3',    # OpenAI
            'CartPoleContinuousBulletEnv-v0', 'InvertedPendulumBulletEnv-v0','InvertedDoublePendulumBulletEnv-v0',      
            'KukaBulletEnv-v0', 'HopperBulletEnv-v0', 'Walker2DBulletEnv-v0',
            'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0', 'HumanoidBulletEnv-v0']

if __name__ == '__main__':
    
    ENV = 4    # select environment
    env_id = gym_envs[ENV]
    env = gym.make(env_id)
    env = env.unwrapped    # allow access to setting env state
    # env._max_episode_steps = 5000
    input_dims = env.observation_space.shape
    num_actions = env.action_space.shape[0]

    warmup = np.array([1 for envs in range(len(gym_envs))]) * 1e3
    warmup[-3:] *= 10

    inputs = {
            # TD3 hyperparameters
            'random_steps': warmup[ENV],    # intial random warmup steps to generate random seed
            'pol_noise': 0.1,               # Gaussian noise added to next agent action
            'targ_pol_noise': 0.2,          # Gaussian noise added to next mini-batch actions
            'targ_pol_clip': 0.5,           # noise cliping of next mini-batch actions

            # SAC hyperparameters
            'r_scale': 1,                   # reward scaling ('inverse temperature')
            's_dist': 'MVN',                # stochastic policy sampling via 'B', 'MVN', 'ST' or 'UVN' distribution

            # execution
            'algo': 'TD3',                  # model 'TD3' or 'SAC'
            'ergodicity': 'Yes',            # assume ergodicity           
            'loss_fn': 'MSE',               # critIc loss 'Cauchy', 'HSC', 'Huber', 'MAE', 'MSE' or 'TCauchy'
            'buffer': 1e6,                  # maximum transistions in experience replay buffer
            'multi_steps': 1,               # bootstrapping of target critic values and rewards
            'discount': 0.99,               # discount factor per step
            'n_trials': 1,                  # number of total trials
            'n_episodes': 1500,             # maximum number of episodes per trial
            'n_cumsteps': 3e6               # maximum cumulative steps per trial regardless of episodes
            }  

    if inputs['algo'] == 'TD3':
        batch_size = 100
        agent = Agent_td3(env_id, env, input_dims, num_actions, lr_alpha=0.001, lr_beta=0.001, tau=0.005, 
                gamma=inputs['discount'], actor_update_interval=2, warmup=inputs['random_steps'], 
                max_size=inputs['buffer'], layer1_dim=400, layer2_dim=300, batch_size=batch_size, 
                policy_noise=inputs['pol_noise'], target_policy_noise=inputs['targ_pol_noise'], 
                target_policy_clip=inputs['targ_pol_clip'], loss_type=inputs['loss_fn'], cauchy_scale=0.420, 
                algo_name=inputs['algo'], erg=inputs['ergodicity'])

    elif inputs['algo'] == 'SAC':
        batch_size = 256
        agent = Agent_sac(env_id, env, input_dims, num_actions, lr_alpha=3e-4, lr_beta=3e-4, lr_kappa=3e-4, 
                tau=0.005, gamma=inputs['discount'], actor_update_interval=1, max_size=inputs['buffer'], 
                layer1_dim=256, layer2_dim=256, batch_size=batch_size, reparam_noise=1e-6, 
                reward_scale=inputs['r_scale'], loss_type =inputs['loss_fn'], cauchy_scale=0.420, 
                stoch=inputs['s_dist'], algo_name=inputs['algo'], erg=inputs['ergodicity'])

    for round in range(inputs['n_trials']):
        best_score = env.reward_range[0]    # set intial best to worst possible reward
        score_history = []
        trial_log = np.zeros((int(inputs['n_episodes']), 9))
        cum_steps = 0
        epis = 0

        # agent.load_models()   # load existing actor-critic network parameters

        while epis < (int(inputs['n_episodes'])): # or cum_steps <= n_cumsteps:
            start_time = time.perf_counter()
            
            state = env.reset()
            done = False

            score = 0
            step = 0

            while not done:
                # action = env.action_space.sample()
                action = agent.select_next_action(state, multi='no')
                next_state, reward, done, info = env.step(action)
                env.state = next_state
                # env.render(mode='human')
                # print(env.state)

                batch_states, batch_actions, batch_rewards, \
                                batch_next_states, batch_dones = agent.get_mini_batch(batch_size)

                batch_targets = agent.multi_step_target(batch_rewards, batch_next_states, batch_dones, 
                                                        env, inputs['multi_steps'], inputs['ergodicity'])

                q1_loss, q2_loss, actor_loss, scale_1, scale_2, logtemp = \
                        agent.learn(batch_states, batch_actions, batch_targets, inputs['loss_fn'], inputs['ergodicity'])

                agent.store_transistion(state, action, reward, next_state, done)    # ensure no sampling of current transition

                # print(next_state)
                env.state = next_state    # reset env to current state when multi-step learning
                # print(env.state)

                state = next_state
                score += reward
                step += 1
                cum_steps += 1

            end_time = time.perf_counter()
            
            score_history.append(score)
            trail_score = np.mean(score_history[-100:])    # 100 episode trailing score
            trial_log[epis, 0], trial_log[epis, 2], trial_log[epis, 8] = end_time - start_time, step, logtemp
            trial_log[epis, 3:6] = np.array([q1_loss, q2_loss, actor_loss])
            trial_log[epis, 6:8] = np.array([scale_1, scale_2])
             
            if trail_score > best_score:
                best_score = trail_score
                agent.save_models()

            print('epis/step/cs {}/{}/{}: score {:1.0f}, trail {:1.0f}, C/A-loss {:1.1f}/{:1.1f}, step/sec {:1.0f}, min {:1.1f}'
                .format(epis+1, step, cum_steps, score, trail_score, 
                        q1_loss+q2_loss, actor_loss, step/(end_time-start_time), (end_time-start_time)/60))
            
            epis += 1

        if not os.path.exists('./results/'+env_id):
            os.makedirs('./results/'+env_id)
        
        dir1 = 'results/'+env_id+'/'+env_id+'_'+inputs['algo']+'_eg'+inputs['ergodicity'][0]+'_'+inputs['loss_fn']
        dir2 = '_b'+str(int(inputs['buffer']/1e6))+'_m'+str(inputs['multi_steps'])
        if inputs['algo'] == 'SAC':
            dir2 += '_rs'+str(inputs['r_scale'])+'_sd'+inputs['sdist']
        dir3 = '_n'+str(inputs['n_trials'])+'_e'+str(int(inputs['n_episodes']))+'_cs'+str(int(inputs['n_cumsteps']/1e6))
        directory = dir1 + dir2 + dir3

        trial_log[:, 1] = np.array(score_history)
        np.save(directory+'.npy', trial_log)

        plot_learning_curve(env_id, inputs, trial_log, directory+'.png')
