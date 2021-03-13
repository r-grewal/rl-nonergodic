import os
import time
from datetime import datetime
import gym
import pybullet_envs
import numpy as np
from algo_td3 import Agent_td3
from algo_sac import Agent_sac
from utils import plot_learning_curve, plot_trial_curve

gym_envs = [# 'LunarLanderContinuous-v2', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3',
            # 'CartPoleContinuousBulletEnv-v0', 'InvertedPendulumBulletEnv-v0',
            # 'InvertedDoublePendulumBulletEnv-v0', 'KukaBulletEnv-v0', 
            'HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', # 'HalfCheetahBulletEnv-v0', 
            'AntBulletEnv-v0', 'HumanoidBulletEnv-v0'
            ]

ENV = 3                     # select environment
env_id = gym_envs[ENV]    
env = gym.make(env_id)
env = env.unwrapped         # allow access to setting enviroment state and remove episode step limit

warmup = np.array([1e3 for envs in range(len(gym_envs))])
warmup[-2:] *= 10

# 'Cauchy', 'CIM', 'HSC', 'Huber', 'MAE', 'MSE', 'MSE2', 'MSE4', 'MSE6', 'TCauchy'
surrogate_critic_loss = ['MSE']

for fn in surrogate_critic_loss:

    inputs = {
            # TD3 hyperparameters
            'policy_noise': 0.1,            # Gaussian exploration noise added to next agent action
            'target_policy_noise': 0.2,     # Gaussian noise added to next target actions
            'target_policy_clip': 0.5,      # noise cliping of next target actions

            # SAC hyperparameters
            'r_scale': 1,                   # reward scaling to offset entropy target ('inverse temperature')
            's_dist': 'UVN',                # stochastic policy via 'LAP' 'MVN', 'ST' or 'UVN' distribution

            # execution details
            'random': warmup[ENV],          # intial random warmup steps to generate random seed
            'discount': 0.99,               # discount factor for successive step
            'trail': 50,                    # moving average count of episode scores for model saving and plots
            'ergodicity': 'Yes',            # assume ergodicity 'Yes' or 'No'  
            'loss_fn': fn,                  # 'Cauchy', 'CIM', 'HSC', 'Huber', 'MAE', 'MSE', 'MSE2', 'MSE4', 'MSE6', 'TCauchy'
            'buffer': 1e6,                  # maximum transistions in experience replay buffer
            'multi_steps': 1,               # bootstrapping of target critic values and rewards
            'n_trials': 3,                  # number of total trials
            'n_cumsteps': 4e4,              # maximum cumulative steps per trial
            'algo': 'SAC'                   # model 'TD3' or 'SAC'
            }  

    if inputs['algo'] == 'TD3':
        batch_size = 100
        agent = Agent_td3(env_id, env, lr_alpha=0.001, lr_beta=0.001, tau=5e-3, layer1_dim=400, layer2_dim=300, 
                          cauchy_scale_1=0.420, cauchy_scale_2=0.420, warmup=inputs['random'], gamma=inputs['discount'], 
                          erg=inputs['ergodicity'], loss_type=inputs['loss_fn'], max_size=inputs['buffer'], 
                          algo_name=inputs['algo'], actor_update_interval=2, batch_size=batch_size,  
                          policy_noise=inputs['policy_noise'], target_policy_noise=inputs['target_policy_noise'], 
                          target_policy_clip=inputs['target_policy_clip'])                

    elif inputs['algo'] == 'SAC':
        batch_size = 256
        agent = Agent_sac(env_id, env, lr_alpha=3e-4, lr_beta=3e-4, lr_kappa=3e-4, tau=5e-3, layer1_dim=256, 
                          layer2_dim=256, cauchy_scale_1=0.420, cauchy_scale_2=0.420, warmup=inputs['random'],
                          gamma=inputs['discount'], erg=inputs['ergodicity'], loss_type=inputs['loss_fn'], 
                          max_size=inputs['buffer'], algo_name=inputs['algo'], actor_update_interval=1, batch_size=batch_size,
                          reparam_noise=1e-6, reward_scale=inputs['r_scale'], stoch=inputs['s_dist'])

    trial_log = np.zeros((inputs['n_trials'], int(inputs['n_cumsteps']), 11))

    for round in range(inputs['n_trials']):
        # agent.load_models()    # load existing actor-critic network parameters
        best_score = env.reward_range[0]    # set intial best to worst possible reward
        score_log = []
        cum_steps, episode = 0, 1

        while cum_steps < int(inputs['n_cumsteps']):
            start_time = time.perf_counter()            
            state = env.reset()
            done = False
            step, score =  0, 0

            while not done:
                # action = env.action_space.sample()
                action, _ = agent.select_next_action(state, inputs['s_dist'], multi='No')
                next_state, reward, done, info = env.step(action)
                env.state = next_state
                # env.render(mode='human')    # render environment visually

                batch_states, batch_actions, batch_rewards, \
                            batch_next_states, batch_dones = agent.get_mini_batch(batch_size)

                batch_targets = agent.multi_step_target(batch_rewards, batch_next_states, batch_dones, 
                                            env, inputs['s_dist'], inputs['ergodicity'], inputs['multi_steps'])

                loss, logtemp, loss_params = agent.learn(batch_states, batch_actions, batch_targets, 
                                                        inputs['loss_fn'], inputs['ergodicity'])

                agent.store_transistion(state, action, reward, next_state, done)    # no current transistion saampling

                env.state = next_state    # reset env to current state when multi-step learning
                state = next_state
                score += reward
                step += 1
                cum_steps += 1

            end_time = time.perf_counter()
            
            trial_log[round, episode-1, 0], trial_log[round, episode-1, 1] =  np.array(end_time-start_time), np.array(score)
            trial_log[round, episode-1, 2], trial_log[round, episode-1, 3:6] = np.array(step), np.array(loss)
            trial_log[round, episode-1, 6], trial_log[round, episode-1, 7:] = np.array(logtemp), np.array(loss_params)

            score_log.append(score)
            trail_score = np.mean(score_log[-inputs['trail']:])
            if trail_score > best_score:
                best_score = trail_score
                agent.save_models()

            fin = datetime.now()

            print('{} {} {:1.0f}/s ep/st/cst {}/{}/{}: r {:1.0f}, r{} {:1.0f}, C/A loss {:1.1f}/{:1.1f}'
                .format(fin.strftime('%d %H:%M:%S'), loss_fn, step/trial_log[round, episode-1, 0], episode, step, 
                        cum_steps, score, inputs['trail'], trail_score, sum(loss[0:2]), loss[2]))

            episode += 1

        if not os.path.exists('./results/'+env_id):
            os.makedirs('./results/'+env_id)
        
        exp = int(len(str(int(inputs['n_cumsteps']))) - 1)
        dir1 = 'results/'+env_id+'/'+env_id+'--'+inputs['algo']+'_g'+inputs['ergodicity'][0]+'_'+inputs['loss_fn']
        dir2 = '_b'+str(int(inputs['buffer']/1e6))+'_m'+str(inputs['multi_steps'])
        if inputs['algo'] == 'SAC':
            dir2 += '_r'+str(inputs['r_scale'])+'_'+inputs['s_dist']
        dir3 = '_'+str(exp)+'s'+str(int(inputs['n_cumsteps']))[0]+'_n'+str(round+1)
        directory = dir1 + dir2 + dir3

        plot_learning_curve(env_id, inputs, trial_log[round], directory+'.png')

    # truncate log up to maximum episodes
    count_episodes = []
    for trial in range(inputs['n_trials']):
        count_episodes.append(np.min(np.where(trial_log[trial, :, 0] == 0)))
    max_episode = np.max(count_episodes) 
    trial_log = trial_log[:, :max_episode, :]

    # plot combined trials
    if inputs['n_trials'] > 1:
        plot_trial_curve(env_id, inputs, trial_log, directory+'_combined.png')

    np.save(directory+'.npy', trial_log)
