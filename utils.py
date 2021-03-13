import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import torch as T
import torch.nn.functional as F

def truncation(estimated, target):
    """
    Elements to be truncated based on Gaussian distribution assumption.

    Paramerters:
        estimated (list): current Q-values
        target (list): Q-values from mini-batch

    Returns:
        estimated (list): truncated current Q-values
        target (list): truncated Q-values from mini-batch
    """
    arg = (target - estimated).detach().clone()
    sigma, mean = T.std_mean(arg, unbiased=False)
    batch_idx = T.where(T.abs(arg - mean) > 3 * sigma)    # 3-sigma rule

    estimated[batch_idx], target[batch_idx] = 0.0, 0.0

    return estimated, target

def cauchy(estimated, target, scale):
    """
    Cauchy loss function.

    Parameters:
        estimated (list): current Q-values
        target (list): Q-values from mini-batch
        scale (float>0): Cauchy scale parameter

    Returns:
        loss (float): loss value
    """    
    arg = ((target-estimated)/scale)**2
    loss = T.log(1 + arg).mean()

    return loss

def nagy_algo(estimated, target, scale):
    """
    Use the Nagy alogrithm to estimate the Cauchy scale paramter based on residual errors.
    
    Parameters:
        estimated (list): current Q-values
        target (list): Q-values from mini-batch
        scale (float>0): current Cauchy scale parameter [step: t]
        
    Returns:
        scale_new: updated scale parameter > 0 [step: t + 1]
    """
    estimated = estimated.detach().clone()
    target = target.detach().clone()
    arg = ((target-estimated)/scale)**2
    arg2 = 1/(1 + arg)
    error = T.sum(arg2) / arg2.shape[0]
    inv_error = 1/error
    
    if inv_error >= 1:
        scale_new = scale * T.sqrt(inv_error - 1).detach().cpu().numpy()
    else:
        scale_new = scale.detach().cpu().numpy()

    return scale_new


def correntropy(estimated, target, kernel):
    """
    Correntropy-Induced Metric (CIM) loss function.

    Parameters:
        estimated (list): current Q-values
        target (list): Q-values from mini-batch
        kernel (float): width of Gaussain

    Returns:
        loss (float): loss value
    """   
    kernel = T.tensor(kernel)
    arg = (target-estimated)**2
    loss = (1 - T.exp(-arg/(2 * kernel**2)) / T.sqrt(2 * np.pi * kernel)).mean() 

    return loss

def cim_size(estimated, target):
    """
    Empirically estimated kernel size for CIM taken as the average reconstruction error.

    Parameters:
        estimated (list): current Q-values
        target (list): Q-values from mini-batch

    Returns:
        kernel (float): standard deviation
    """
    arg = (target-estimated)**2
    kernel = arg.detach().clone().mean()

    return kernel.cpu().numpy()

def hypersurface(estimated, target):
    """
    Hypersurface cost based loss function.

    Parameters:
        estimated (list): current Q-values
        target (list): Q-values from mini-batch

    Returns:
        loss (float): loss value
    """    
    arg = (target-estimated)**2
    loss = (T.sqrt(1 + arg) - 1).mean() 

    return loss

def mse(estimated, target, exp=0):
    """
    MSE loss function.

    Parameters:
        estimated (list): current Q-values
        target (list): Q-values from mini-batch
        exp (even integer): exponent in addition to MSE

    Returns:
        loss (float): loss value
    """
    arg = (target-estimated)**(2 + int(exp))
    loss = arg.mean()

    return loss

def loss_function(estimated, target, loss_type, scale, kernel):
    """
    Use the Nagy alogrithm to estimate the Cauchy scale paramter based on residual errors.
    
    Parameters:
        estimated (list): current Q-values
        target (list): Q-values from mini-batch
        loss_type (str): alphabetised loss functions
        scale (float>0): current Cauchy scale parameter
        kernel (float): standard deviation for CIM 

    Returns:
        loss (float): loss value
    """
    if loss_type == "Cauchy":
        loss = cauchy(estimated, target, scale)
    elif loss_type == "CIM":
        loss = correntropy(estimated, target, kernel)
    elif loss_type == "HSC":
        loss = hypersurface(estimated, target)
    elif loss_type == "Huber":
        loss = F.smooth_l1_loss(estimated, target)
    elif loss_type == "MAE":
        loss = F.l1_loss(estimated, target)
    elif loss_type == "MSE":
        loss = F.mse_loss(estimated, target)
    elif loss_type == "MSE2":
        loss = mse(estimated, target, 2)
    elif loss_type == "MSE4":
        loss = mse(estimated, target, 4)
    elif loss_type == "MSE6":
        loss = mse(estimated, target, 6)
    elif loss_type =="TCauchy":
        estimated, target = truncation(estimated, target)
        loss = cauchy(estimated, target, scale)

    return loss
    
def plot_learning_curve(env_id, input_dict, trial_log, filename_png):
    """
    Plot of game running average score and critic loss for environment.
    
    Parameters:
        env_id (str): name of environment
        input_dict (dict): dictionary of all execution details
        trial_log (array): log of episode data of a single trial
        filename_png (directory): save path of plot
    """
    # truncate log up to maximum episodes
    try:
        trial_log = trial_log[:np.min(np.where(trial_log[:, 0] == 0))]
    except:
        pass
    
    score_log = trial_log[:, 1]
    steps = trial_log[:, 2]
    critic_log = trial_log[:, 3:5].sum(axis=1)

    # ignore intial NaN critic loss when batch_size > buffer
    idx, loss = 0, 0
    while np.nan_to_num(loss) == 0:
        loss = critic_log[idx]
        idx += 1

    offset = np.max(idx - 1, 0)
    score_log = score_log[offset:]
    steps = steps[offset:]
    critic_log = critic_log[offset:]
    length = len(score_log)

    # obtain cumulative steps for x-axis
    cum_steps = np.zeros(length)
    cum_steps[0] = steps[0]
    for i in range(length-1):
        cum_steps[i+1] = steps[i+1] + cum_steps[i]

    exp = int(len(str(int(np.max(cum_steps)))) - 1)
    x_steps = cum_steps/10**(exp)
    
    # calculate moving averages
    trail = input_dict['trail']
    running_avg1 = np.zeros(length)
    for i in range(length-offset):
        running_avg1[i+offset] = np.mean(score_log[max(0, i-trail):(i+1)])

    running_avg2 = np.zeros(length)
    for i in range(length-offset):
        running_avg2[i+offset] = np.mean(critic_log[max(0, i-trail):(i+1)])

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, label='score')
    ax2 = fig.add_subplot(1,1,1, label='critic', frame_on=False)

    ax1.plot(x_steps, running_avg1, color='C0')
    ax1.set_xlabel('Training Steps (1e'+str(exp)+')')
    ax1.yaxis.tick_left()
    ax1.set_ylabel('Average Score', color='C0')
    ax1.yaxis.set_label_position('left')
    ax1.tick_params(axis='y', colors='C0')
    ax1.grid(True, linewidth=0.5)

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    ax1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    ax2.plot(x_steps, running_avg2, color='C3')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Average Critic Loss', color='C3')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C3')
    
    # make vertical lines splitting fractions of total episodes
    partitions = 0.25
    for block in range(int(1/partitions-1)):
        period = x_steps[int(np.min(length * partitions * (block + 1)))-1]
        ax1.vlines(x=period, ymin=ymin, ymax=ymax, linestyles ="dashed", color='C2')

    # make vertical line when TD3 begins learning
    # if input_dict['algo'] == 'TD3':
    #     warmup = x_steps[np.min(np.where(cum_steps - input_dict['random'] > 0))]
    #     ax1.vlines(x=warmup, ymin=ymin, ymax=ymax, linestyles ="dashed", color='C7')

    tit1 = 'Trailing '+str(int(input_dict['trail']))+' Episode Averages and '+str(partitions)[2:4]+'% Partitions \n'
    tit2 = input_dict['algo']+': \''+env_id+'\' '+'('+'g'+input_dict['ergodicity'][0]+', '+input_dict['loss_fn']+', '
    tit3 = 'b'+str(input_dict['buffer']/1e6)[0]+', '+'m'+str(input_dict['multi_steps'])+', '
    if input_dict['algo'] == 'SAC':
            tit3 += 'r'+str(input_dict['r_scale'])+', '+input_dict['s_dist']+', '
    tit4 = 'e'+str(int(length))+')'
    title = tit1 + tit2 + tit3 + tit4

    ax1.set_title(title)

    plt.savefig(filename_png, bbox_inches='tight', dpi=600, format='png')

def plot_trial_curve(env_id, input_dict, trial_log, filename_png):
    """
    Plot of interpolated mean and MAD score and critic loss across all trials for environment.
    
    Parameters:
        env_id (str): name of environment
        input_dict (dict): dictionary of all execution details
        trial_log (array): log of episode data of a single trial
        filename_png (directory): save path of plot
    """
    score_log = trial_log[:, :, 1]
    steps_log = trial_log[:, :, 2]
    critic_log = trial_log[:, :, 3:5].sum(axis=2)

    # find maximum number of episodes in each trial
    max_episodes = []
    for trial in range(steps_log.shape[0]):
        try:
            max_episodes.append(np.min(np.where(steps_log[trial, :] == 0)))
        except:
            max_episodes.append(steps_log.shape[1])
    
    # ignore intial NaN critic loss when batch_size > buffer
    offset = []
    for trial in range(steps_log.shape[0]):
        idx, loss = 0, 0

        while np.nan_to_num(loss) == 0:
            loss = critic_log[trial, idx]
            idx += 1

        offset.append(idx)

    max_offset = np.maximum(np.array(offset) - 1, 0)
    length = steps_log.shape[1] - np.min(max_offset)
    
    scores = np.zeros((steps_log.shape[0], length))
    steps = np.zeros((steps_log.shape[0], length))
    critics = np.zeros((steps_log.shape[0], length))

    for trial in range(steps.shape[0]):
        scores[trial, :length - max_offset[trial]] = score_log[trial, max_offset[trial]:]
        steps[trial, :length - max_offset[trial]] = steps_log[trial, max_offset[trial]:]
        critics[trial, :length - max_offset[trial]] = critic_log[trial, max_offset[trial]:]

    # obtain cumulative steps for x-axis for each trial
    cum_steps = np.zeros((steps.shape[0], length))
    cum_steps[:, 0] = steps[:, 0]
    for trial in range(steps.shape[0]):
        for e in range(max_episodes[trial]-max_offset[trial]-1):
            cum_steps[trial, e+1] = steps[trial, e+1] + cum_steps[trial, e]
    
    exp = int(len(str(int(input_dict['n_cumsteps']) - 1)))
    x_steps = cum_steps/10**(exp)   

    # create lists for interteploation
    list_steps = []
    list_scores = []
    list_critic = []
    for trial in range(scores.shape[0]):
        trial_step = []
        trial_score = []
        trial_critic = []
        for epis in range(max_episodes[trial]-max_offset[trial]):
            trial_step.append(x_steps[trial, epis])
            trial_score.append(scores[trial, epis])
            trial_critic.append(critics[trial, epis])
        list_steps.append(trial_step)
        list_scores.append(trial_score)
        list_critic.append(trial_critic)

    # linearly interpolate mean and MAD across trials
    count_x = list_steps[max_episodes.index(max(max_episodes))]
    score_interp = [np.interp(count_x, list_steps[i], list_scores[i]) for i in range(steps.shape[0])]
    critic_interp = [np.interp(count_x, list_steps[i], list_critic[i]) for i in range(steps.shape[0])]

    score_mean = np.mean(score_interp, axis=0)
    score_mad = np.mean(np.abs(score_interp - score_mean), axis=0)
    critic_mean = np.mean(critic_interp, axis=0)
    critic_mad = np.mean(np.abs(critic_interp - critic_mean), axis=0)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, label='score')
    ax2 = fig.add_subplot(1,1,1, label='critic', frame_on=False)

    ax1.plot(count_x, score_mean, color='C0')
    ax1.fill_between(count_x, score_mean-score_mad, score_mean+score_mad, facecolor='C0', alpha=0.4)
    ax1.set_xlabel('Training Steps (1e'+str(exp)+')')
    ax1.yaxis.tick_left()
    ax1.set_ylabel('Score', color='C0')
    ax1.yaxis.set_label_position('left')
    ax1.tick_params(axis='y', colors='C0')
    ax1.grid(True, linewidth=0.5)

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    ax1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    ax2.plot(count_x, critic_mean, color='C3')
    ax2.fill_between(count_x, critic_mean-critic_mad, critic_mean+critic_mad, facecolor='C3', alpha=0.4)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Critic Loss', color='C3')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C3')

    tit1 = 'Linearly Interpolated Mean and MAD Bands of '+str(input_dict['n_trials'])+' Trials \n'
    tit2 = input_dict['algo']+': \''+env_id+'\' '+'('+'g'+input_dict['ergodicity'][0]+', '+input_dict['loss_fn']+', '
    tit3 = 'b'+str(input_dict['buffer']/1e6)[0]+', '+'m'+str(input_dict['multi_steps'])+', '
    if input_dict['algo'] == 'SAC':
            tit3 += 'r'+str(input_dict['r_scale'])+', '+input_dict['s_dist']+', '
    tit4 = 'e'+str(int(length))+')'
    title = tit1 + tit2 + tit3 + tit4

    ax1.set_title(title)
    
    plt.savefig(filename_png, bbox_inches='tight', dpi=600, format='png')
