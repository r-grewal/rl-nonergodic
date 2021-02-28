import gym
import pybullet_envs
import numpy as np
from algo_td3 import Agent_td3
from algo_sac import Agent_sac
from utils import plot_learning_curve

# OpenAI gym environments
# 'LunarLanderContinuous-v2', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3'

# PyBullet gym environments 
# 'CartPoleContinuousBulletEnv-v0', 'InvertedPendulumBulletEnv-v0', 'Walker2DBulletEnv-v0',
# 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0', 'HumanoidBulletEnv-v0'

if __name__ == '__main__':

    envs = ['LunarLanderContinuous-v2', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3',
            'CartPoleContinuousBulletEnv-v0', 'InvertedPendulumBulletEnv-v0', 'Walker2DBulletEnv-v0',
            'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0', 'HumanoidBulletEnv-v0']

    env_id = envs[0]

    env = gym.make(env_id)
    input_dims = env.observation_space.shape
    num_actions = env.action_space.shape[0]

    if env_id == envs[0] or env_id == envs[1] or env_id == envs[2]:
        env._max_episode_steps = 2500

    algo = 'TD3'            # set to 'TD3' or 'SAC'
    loss_fn = 'MSE'         # set to 'Cauchy', 'HSC', 'Huber', 'MAE', 'MSE' or 'TCauchy'
    random_steps = 1000     # intial random steps for TD3
    buffer = 1e6            # size of experience replay buffer

    if algo == 'TD3':
        agent = Agent_td3(env_id, env, input_dims, num_actions, lr_alpha=0.001, lr_beta=0.001, tau=0.005, 
                gamma=0.99, actor_update_interval=2, warmup=random_steps, max_size=buffer, layer1_dim=400, 
                layer2_dim=300, batch_size=100, policy_noise=0.1, target_policy_noise=0.2, 
                target_policy_clip=0.5, loss_type=loss_fn, cauchy_scale=0.420, algo_name=algo)

    elif algo == 'SAC':
        agent = Agent_sac(env_id, env, input_dims, num_actions, lr_alpha=3e-4, lr_beta=3e-4, lr_kappa=3e-4, 
                tau=0.005, gamma=0.99, actor_update_interval=1, max_size=buffer, layer1_dim=256, layer2_dim=256, 
                batch_size=256, reparam_noise=1e-6, reward_scale=1, loss_type =loss_fn, cauchy_scale=0.420, algo_name=algo)

    n_rounds = 1            # number of trials
    n_games = 1000          # number of episodes per trial

    directory = 'results/'+env_id+'/'+env_id+'_'+algo+'_'+loss_fn+'_'+'e'+str(n_games)+'_'+'n'

    for itr in range(n_rounds):
        # set intial best to worst possible reward
        best_score = env.reward_range[0]
        score_history = []
        step_history = np.zeros((n_games,), dtype=np.uint32)
        loss_history = np.zeros((n_games, 3))
        scale_history = np.zeros((n_games, 2))

        if algo == 'SAC':
            logtemp_history = np.zeros((n_games,))

        for i in range(n_games):
            state = env.reset()
            done = False

            score = 0
            step = 0
            # agent.load_models()

            while not done:
                # action = env.action_space.sample()
                action = agent.select_next_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transistion(state, action, reward, next_state, done)
                q1_loss, q2_loss, actor_loss, scale_1, scale_2, logtemp = agent.learn(loss_type=loss_fn)
                state = next_state

                score += reward
                step += 1
                # print(step)
                # env.render(mode='human')

            score_history.append(score)
            trail_score = np.mean(score_history[-100:])
            step_history[i] = step
            loss_history[i] = np.array([q1_loss, q2_loss, actor_loss])
            scale_history[i] = np.array([scale_1, scale_2])
            
            if algo == 'SAC':
                logtemp_history[i] = logtemp

            print('epis/step {}/{}: score {:1.1f}, trail {:1.1f}, C/A loss {:1.1f}/{:1.1f}'
                .format(i+1, step, score, trail_score, q1_loss+q2_loss, actor_loss))

            if trail_score > best_score:
                best_score = trail_score
                agent.save_models()

        filename_npy_score = directory + str(itr+1)+'_score'+'.npy'
        filename_npy_step = directory + str(itr+1)+'_step'+'.npy'
        filename_npy_loss = directory + str(itr+1)+'_loss'+'.npy'
        filename_npy_scale = directory + str(itr+1)+'_scale'+'.npy'
        filename_npy_logtemp = directory + str(itr+1)+'_temp'+'.npy'
        filename_png = directory + str(itr+1)+'.png'
        
        score_history = np.array(score_history)
        np.save(filename_npy_score, score_history)
        np.save(filename_npy_step, step_history)
        np.save(filename_npy_loss, loss_history)
        np.save(filename_npy_scale, scale_history)
        
        if algo == 'SAC':
            np.save(filename_npy_logtemp, logtemp_history)

        plot_learning_curve(score_history, filename_png)