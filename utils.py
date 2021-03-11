from typing import runtime_checkable
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import partition
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
    arg = target.detach().clone() - estimated.detach().clone()
    mean, sigma = T.mean(arg), T.std(arg)
    batch_idx = T.where(T.abs(arg - mean) > 3*sigma)

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
        scale_new = scale

    return scale_new


def correntropy(estimated, target):
    """
    Correntropy-Induced Metric loss functions with empirically estimated kernel size 
    taken as the average reconstruction error.

    Parameters:
        estimated (list): current Q-values
        target (list): Q-values from mini-batch

    Returns:
        loss (float): loss value
    """   
    arg = (target-estimated)**2
    kernel = arg.detach().clone().mean()
    loss = (1 - T.exp(-arg/(2 * kernel**2)) / T.sqrt(2 * np.pi * kernel)).mean() 

    return loss

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

    arg = (target-estimated)**(2 + exp)
    loss = arg.mean()

    return loss

def loss_function(estimated, target, loss_type, scale):
    """
    Use the Nagy alogrithm to estimate the Cauchy scale paramter based on residual errors.
    
    Parameters:
        estimated (list): current Q-values
        target (list): Q-values from mini-batch
        loss_type (str): alphabetised loss functions
        
    Returns:
        loss (float): loss value
    """
    if loss_type == "Cauchy":
        loss = cauchy(estimated, target, scale)
    elif loss_type == "CIM":
        loss = correntropy(estimated, target)
    elif loss_type == "HSC":
        loss = hypersurface(estimated, target)
    elif loss_type == "Huber":
        loss = F.smooth_l1_loss(estimated, target)
    elif loss_type == "MAE":
        loss = F.l1_loss(estimated, target)
    elif loss_type == "MSE":
        loss = F.mse_loss(estimated, target)
    elif loss_type =="TCauchy":
        estimated, target = truncation(estimated, target)
        loss = cauchy(estimated, target, scale)
    elif loss_type == "X4":
        loss = mse(estimated, target, 2)
    elif loss_type == "X6":
        loss = mse(estimated, target, 4)
    elif loss_type == "X8":
        loss = mse(estimated, target, 6)

    return loss
    
def plot_learning_curve(env_id, input_dict, trial_log, filename_png):
    """
    Plot of game running average score and critic loss for environment.
    
    Parameters:
        env_id (str): name of environment
        input_dict (dict): dictionary of all execution details
        trial_log (array): log of episode data
        filename_png (directory): save path of plot
    """
    score_log = trial_log[:, 1]
    length = len(score_log)
    critic_log = trial_log[:, 3:5].sum(axis=1)

    # obtain cumulative steps for x-axis
    steps = trial_log[:, 2]
    cum_steps = np.zeros(length)
    cum_steps[0] = steps[0]
    for i in range(length-1):
        cum_steps[i+1] = steps[i+1] + cum_steps[i]
    
    exp = int(len(str(int(np.max(cum_steps)))) - 1)
    x_steps = np.round(cum_steps/10**(exp), 1)
    
    # ignore intial NaN critic loss when batch_size > buffer
    idx, loss = 0, 0
    while np.nan_to_num(loss) == 0:
        loss = critic_log[idx]
        idx += 1

    offset = np.max(idx - 1, 0)
    score_log = score_log[offset:] 
    critic_log = critic_log[offset:]

    # calculate moving averages
    trail = input_dict['trail']
    running_avg1 = np.zeros(length, dtype=np.int)
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
    #     warmup = x_steps[np.min(np.where(cum_steps - input_dict['random_steps'] > 0))]
    #     ax1.vlines(x=warmup, ymin=ymin, ymax=ymax, linestyles ="dashed", color='C7')

    ax1.set_title('Trailing '+str(int(input_dict['trail']))+' Episode Averages and '+
                str(partitions)[2:4]+'% Partitions \n'+
                input_dict['algo']+': \''+env_id+'\' '+
                '('+'g'+input_dict['ergodicity'][0]+', '+input_dict['loss_fn']+', '+
                'b'+str(input_dict['buffer']/1e6)[0]+', '+'m'+str(input_dict['multi_steps'])+
                ', ''e'+str(int(length))+')')

    plt.savefig(filename_png, bbox_inches='tight', dpi=600, format='png')
