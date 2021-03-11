import os
import time
import datetime
import gym
import pybullet_envs
import numpy as np
from algo_td3 import Agent_td3
from algo_sac import Agent_sac
from utils import plot_learning_curve

gym_envs = [# 'LunarLanderContinuous-v2', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3',    # OpenAI envs
            # 'CartPoleContinuousBulletEnv-v0', 'InvertedPendulumBulletEnv-v0',    # PyBullet envs
            # 'InvertedDoublePendulumBulletEnv-v0', 'KukaBulletEnv-v0', 
            'HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 
            'AntBulletEnv-v0', 'HumanoidBulletEnv-v0']
warmup = np.array([1e3 for envs in range(len(gym_envs))])
warmup[-3:] *= 10

# 'Cauchy', 'CIM', 'HSC', 'Huber', 'MAE', 'MSE', 'TCauchy', 'X4', 'X6'
for loss in ['Cauchy', 'CIM', 'HSC', 'Huber', 'MAE', 'MSE', 'TCauchy', 'X4', 'X6']:
    if __name__ == '__main__':
        
        ENV = 1                         # select environment
        env_id = gym_envs[ENV]          
        env = gym.make(env_id)          # create environment
        env = env.unwrapped             # allow access to setting enviroment state and remove episode step limit

        inputs = {
                # TD3 hyperparameters
                'random_steps': warmup[ENV],    # intial random warmup steps to generate random seed
                'pol_noise': 0.1,               # Gaussian noise added to next agent action
                'targ_pol_noise': 0.2,          # Gaussian noise added to next mini-batch actions
                'targ_pol_clip': 0.5,           # noise cliping of next mini-batch actions

                # SAC hyperparameters
                'r_scale': 1,                   # reward scaling ('inverse temperature')
                's_dist': 'MVN',                # stochastic policy via 'B', 'LAP' 'MVN', 'ST' or 'UVN' distribution

                # execution details
                'discount': 0.99,               # discount factor per step
                'trail': 50,                    # moving average count of episode scores for model saving and plots
                'algo': 'TD3',                  # model 'TD3' or 'SAC'
                'ergodicity': 'Yes',            # assume ergodicity 'Yes' or 'No'  
                'loss_fn': loss,                # critic loss 'Cauchy', 'CIM', 'HSC', 'Huber', 'MAE', 'MSE', 'TCauchy', 'X4/6/8'
                'buffer': 1e6,                  # maximum transistions in experience replay buffer
                'multi_steps': 1,               # bootstrapping of target critic values and rewards
                'n_trials': 3,                  # number of total trials
                'n_cumsteps': 2e5               # maximum cumulative steps per trial regardless of episodes
                }  

        input_dims = env.observation_space.shape    # input dimensions tuple
        num_actions = env.action_space.shape[0]     # number of possible actions

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
            time_log, score_log, step_log, logtemp_log, loss_log, scale_log = [], [], [], [], [], []
            cum_steps = 0
            epis = 0
            
            # agent.load_models()   # load existing actor-critic network parameters

            while cum_steps < int(inputs['n_cumsteps']):
                start_time = time.perf_counter()
                
                state = env.reset()
                done = False

                score = 0
                step = 0

                while not done:
                    # action = env.action_space.sample()
                    action, _ = agent.select_next_action(state, inputs['s_dist'], multi='No')
                    next_state, reward, done, info = env.step(action)
                    # env.render(mode='human')    # render environment visuals
                    env.state = next_state

                    batch_states, batch_actions, batch_rewards, \
                                    batch_next_states, batch_dones = agent.get_mini_batch(batch_size)

                    batch_targets = agent.multi_step_target(batch_rewards, batch_next_states, batch_dones, 
                                                            env, inputs['s_dist'], inputs['ergodicity'], inputs['multi_steps'])

                    q1_loss, q2_loss, actor_loss, scale_1, scale_2, logtemp = \
                                agent.learn(batch_states, batch_actions, batch_targets, inputs['loss_fn'], inputs['ergodicity'])

                    agent.store_transistion(state, action, reward, next_state, done)    # ensure no current transistion sampling

                    env.state = next_state    # reset env to current state when multi-step learning
                    # print(next_state, done)
                    state = next_state
                    score += reward
                    step += 1
                    cum_steps += 1

                end_time = time.perf_counter()
                fin = datetime.datetime.now()
                
                time_log.append(end_time - start_time)
                score_log.append(score)
                step_log.append(step)
                loss_log.append([q1_loss, q2_loss, actor_loss])
                scale_log.append([scale_1, scale_2])
                logtemp_log.append(logtemp)

                trail_score = np.mean(score_log[-inputs['trail']:])
                if trail_score > best_score:
                    best_score = trail_score
                    agent.save_models()

                print('{} {:1.0f}/s ep/st/cst {}/{}/{}: r {:1.0f}, r{} {:1.0f}, C/A loss {:1.1f}/{:1.1f}'
                    .format(fin.strftime('%d %H:%M:%S'), step/time_log[-1], epis+1, step, cum_steps, 
                            score, inputs['trail'], trail_score, q1_loss+q2_loss, actor_loss))
                
                epis += 1

            if not os.path.exists('./results/'+env_id):
                os.makedirs('./results/'+env_id)
            
            exp = int(len(str(int(inputs['n_cumsteps']))) - 1)
            dir1 = 'results/'+env_id+'/'+env_id+'--'+inputs['algo']+'_g'+inputs['ergodicity'][0]+'_'+inputs['loss_fn']
            dir2 = '_b'+str(int(inputs['buffer']/1e6))+'_m'+str(inputs['multi_steps'])
            if inputs['algo'] == 'SAC':
                dir2 += '_r'+str(inputs['r_scale'])+'_d'+inputs['sdist']
            dir3 = '_'+str(exp)+'s'+str(int(inputs['n_cumsteps']))[0]+'_n'+str(round+1)
            directory = dir1 + dir2 + dir3

            trial_log = np.zeros((len(score_log), 9))
            trial_log[:, 0], trial_log[:, 1], trial_log[:, 2]  = np.array(time_log), np.array(score_log), np.array(step_log)
            trial_log[:, 3:6], trial_log[:, 6:8], trial_log[:, 8] = np.array(loss_log), np.array(scale_log), np.array(logtemp_log)

            np.save(directory+'.npy', trial_log)
            plot_learning_curve(env_id, inputs, trial_log, directory+'.png')
