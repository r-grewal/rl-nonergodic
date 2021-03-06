# Project Log

* 2021-03-16

Automatic temperature scaling is working correctly but ended up slowing the speed of learning tremendously. This was because I set the reparameterisation noise down to 1e-8 from 1e-6. This was hoping to achieve greater accuracy in log probabilities of actions as they are summed. However, this was fatal error and significantly recued the rate of learning to become completely impractical, similarly 1e-7 is also not feasible, while 1e-6 and above seem to work. This is very strange but as seen in the links provided in networks_sac.py function stochastic_uv() several other working repositories, it is necessary. This minute addition seems to become important when the probability of an action increases. Several other repositories using TensorFlow 2 do not require such a correction, perhaps this due then to the PyTorch torch.log() function.

All explicit references to reward scaling factor have been removed as it is no longer necessary since temperature is being automatically scaled.


* 2021-03-15

Realised that automatic temperature tuning was not taking place and instead we were working with fixed value of 1 as with earlier SAC work. Corrected with minor modification. Temperature behaviour is found to be consistent with literature where it quickly crashes to zero and steadily begins to climb.

Humanoid movement experiment with MSE conducted and is seen to be a very tough challenge. TD3 begins learning whereas as alluded to in update 2021-03-03, SAC struggles when dealing with negative episodic scores. This is because the environment is so challenging that the agent consistently fails for a very long time while -1 penalties per time step accumulate. Then being off-policy, when the mini-batch is pooled the agent has no concept of positive rewards and only maximises to zero. Some confirmation of this issue is found in [Raffin and Stulp (2020)](https://arxiv.org/pdf/2005.05719.pdf) where they also did not run humanoid experiments when it would have simply involved typing “HumanoidBulletEnv-v0” once.

There doesn’t seem to be any general solution to this problem. While we could add a constant score offset for first couple thousand episodes to ensure non-negative scores, the same would have to be done for TD3. For the time being we will not run anymore humanoid experiments.

See results for trial humanoid runs. It is empirically found that TD3 is roughly 1.5x faster than SAC the exact source of the difference is unclear but might be due to the 2.5x large mini-batch size for SAC.

*Mini-batch size has no real impact on speed after running several benchmarks. Student-t distribution has been removed as selecting the correct degrees of freedom is non-trivial.*

* 2021-03-14

Learned how RAM works and that appending is much faster than inputting into a large numpy array. Modified main.py to append episode data resulting in 8-11% increase in overall speed.

For several environments either at the early or late stages episodes are likely to run for many steps. For the early stage this is not really a problem, for the later stages however, having 300k+ steps in an episode can cause significant annoyances since this can increase the pre-set maximum number of steps by a huge amount. While this a great sign, i.e. the AI is learning to walk for extended periods of time, the Pybullet environments are not really the focus of this project. To simply evaluate robust twin critic loss we need only comparative learning curves for different loss functions. As such, we limit the maximum number of steps. 

Letting the AI continue playing the game indefinitely till it losses will be crucial when evaluating risk-taking/management simulations since here we actually desire it to run forever i.e. not go bankrupt.

* 2021-03-13

Fixed catastrophic error in main.py. In each trial the neural networks were not being reset and so training was effectively one large trail. While the previous TD3 are still valid, they essentially represent one 900k step run. These will be removed for clarity. Ultimately this has very significantly reduced run-to-run variance.

SAC implemented and (mostly) working again. Improved output plots to be more smooth, add gridlines and includes combined all-trial plot with interpolated mean and MAD. Minor changes to Cauchy scale and kernel size requires re-runs of all experiments just to be consistent.

SAC implementation works for 3 actor policy sampling distributions: normal, student-t and Laplace. Student-t can facilitate an analysis on the impact of number of actions batch size on final result through the degrees of freedom parameter and whether scaling effects the result. Laplace will allow analysis on how important agent variance/exploration is since it is significantly more likely to sample near the mean. Multivariable normal has a bug that cause the neural network to yield NaN outputs for steps (roughly) > batch size, the cause of this error is unknown for the time being. Also modified algorithm to add random initial steps to ensure a new seed is generated each trial. 

Overall, majority of code is seemingly complete. Still need to get multi-step returns code functioning for both algorithms. Incorporating non-ergodicity is fairly simple but is only possible if the sum of rewards and target Q-values is non-negative, a straightforward implementation would be to only enable non-ergodic returns after learning begins to occur (however this defeats the point of using a non-ergodic return).

Humanoid experiments will be run next for SAC and uploaded when complete. HalfCheetah experiments will also no longer be run as it is not distinct enough to be worthy of the computational run-time. This will free up resources to run more risk-taking/management experiments later on.


* 2021-03-10

Preliminary TD3 200k step runs with 3 trials for Hopper and Walker2D reveal very interesting results. We visually observe the opposite of what was expected with different surrogate loss functions. For example, in traditional supervised ML, L1 loss performs internal feature selection at the cost of stability. Similarly, the other loss functions perform differing levels of outlier selection and removal either through dampening their magnitude or explicit truncation. 

For the Hopper environment we find that MSE appears superior. More precisely, in the early post-warmup period for TD3, critic loss functions that amplify large values appear necessary for longer term learning. In other words, repeatedly making large mistakes early on seems crucial for future success. Or alternatively, that allowing for relatively massive early failure is key to learning.

We see this on display with the success of the other loss functions inversely proportion to their strength. Interestingly this leads to Truncated Cauchy performing even worse than Cauchy which was absolutely not expected. To further test this we introduce two additional loss functions, the correntropy-induced metric loss which provides even greater outlier smoothing than Cauchy,  and use higher even powers of MSE up to MSE^4 as way to extremely amplify losses to see whether it learns even faster.

Correntropy, as expected performs even worse in reduces learning. MSE^2 (X4) was pretty good, MSE^4 (X6) was functional but somewhat reduced performance, however was still vastly superior to Cauchy and correntropy. Therefore even if Cauchy scale and CIM kernel size estimation is wrong, the MSE^(2n) results are clear. This provides further weight to our hypothesise. This is very strange as amplification of outliers is generally not known to result in robust models.

Colloquially, we can say that repeated failure is found to be crucial to long-terms success. This is most likely due to the large losses backprop yielding optimal weights at a crucial time. This is interesting that even though we are sampling uniformly form the buffer, we must achieve the correct weights promptly before the buffer is filled by garbage. Perhaps using prioritised experience replay would be better.
 
Overall, 1/3 key investigations of the project have been performed and are subject to corrections after more lengthy runs on Artemis. Key finding: unlike supervised or unsupervised ML, amplifying the effects of outliers appears highly beneficial to continuous action-space RL. This is polar opposite of what we expected but makes some sense in hindsight (a theoretical explanation will be written in TeX soon).

Furthermore, preliminary results indicate the number of environments to be tested can be reduced without loss of generality. The OpenAI gym environments will be omitted as they are computationally inefficient and quite simple to solve using existing models. For Pybullet, CartPole, InvertedPendulum InvertedDoublePendulum will similarly be removed, Kuka will be removed as it presents some instability with regards to multiple runs occasionally causing the physics engine to crash. Therefore, only the five most challenging environments will be tested.

SAC linked to main.py but doesn’t work correctly after making modular improvements for the time being.


* 2021-03-06

Link to SAC from main.py is broken and needs eventual updating. TD3 single-step learning fully functioning again, multi-step learning code runs but agent does not seem to learn in any reasonable time due to bootstrapped target Q-values not being backpropagated correctly. This is likely due to the formats of the torch tensors and grads not being correct.

*The multi-step issue is unlikely due to torch and more likely a result of the gym environment object not being initialised correctly. Multi-step also cripples GPU performance with 2-steps reducing throughput by 13x due to for loops. Since getting up to 1e7 transitions into the buffer is going take long time, multi-step will therefore likely be approached again in the later part of the project.*

* 2021-03-04

Incorporated multi-step learning into TD3 requiring major refactoring. Framework for doing the same for SAC along with allowing the use of different stochastic sampling distributions outlined. Compacted output logs to just one numpy array. Updated to torch 1.8 w/ cuda 11.1 which should provide large speed boost on RTX 3070 and pybullet 3.0.9. TD3 Agent now completely fails to learn after 1000 initial random steps with the source of error unknown.

*Corrected by refactoring TD3 code to be more modular and assigning default torch tensors for self.select_next_action().*

* 2021-03-03

TD3 appears to be well-optimised and functioning. Ready to parallelise across multiple GPUs on Artemis HPC to run 10 trials, each of 3e6 cumulative steps per environment per loss function.

SAC appears to functioning correctly but appears to struggle in OpenAI environments where large negative scores occur causing the algorithm to maximise only up to zero. This is likely due to the replay buffer containing minimal positive scores and hence algorithm fails to learn multi-modal solutions, no issues appear to occur with PyBullet environments. Regardless, code is almost ready to parallelise to run on Artemis HPC.

Additionally, for SAC our use of true multi-dimensional stochastic Gaussian noise added appears to heavily reduce performance compared the simpler Gaussian sampling used by the overwhelming majority of implementations. Our method allows each sample state contained in the mini-batch to generate its own covariance matrix and hence each action dimension will have unique variance. We expect this to significantly improve the actor policy for complex environments. However the trade-off between learning in a fewer number of steps and each of step taking longer is unclear and environment dependent.

* 2021-03-01

SAC currently requires tuning to transfer high CPU usage (8 threads at 100% utilisation) to the GPU. Algorithm also appears for certain environments to struggle in learning multi-modal solutions. Additionally, our use of true multi-dimensional stochastic Gaussian noise added to each policy action component inside every sample contained the mini-batch appears to reduce performance compared the simpler Gaussian sampling used by the overwhelming majority of available implementations.

*Fixed by sending stochastic sampling from distributions to the GPU.*
