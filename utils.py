import numpy as np
import matplotlib.pyplot as plt
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
    loss = T.log(1 + arg).sum()

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
    loss = (T.sqrt(1 + arg) - 1).sum()

    return loss

def loss_function(estimated, target, loss_type, scale):
    """
    Use the Nagy alogrithm to estimate the Cauchy scale paramter based on residual errors.
    
    Parameters:
        estimated (list): current Q-values
        target (list): Q-values from mini-batch
        loss_type (str): Cauchy, HSC, Huber, MAE, MSE, TCauchy loss functions
        
    Returns:
        loss (float): loss value
    """
    if loss_type == "Cauchy":
        loss = cauchy(estimated, target, scale)
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

    return loss
    
def plot_learning_curve(scores, filename_png):
    """
    Plot of 100 game running average for environment.
    
    Parameters:
        scores (list): list of final scores of each episode
        filename_png (directory): save path of plot
    """
    running_avg = np.zeros(len(scores))
    x = [i+1 for i in range(len(scores))]

    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])

    plt.plot(x, running_avg)
    plt.title('Moving average of trailing 100 episodes')
    plt.savefig(filename_png)
