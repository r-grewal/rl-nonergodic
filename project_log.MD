# Project Log

* 2021-03-06

Link to SAC from main.py is broken and needs eventual updating. TD3 single-step learning fully functioning again, multi-step learning code runs but agent does not seem to learn in any reasonable time due to bootstrapped target Q-values not being backpropagated correctly. This is likely due to the formats of the torch tensors and grads not being correct.

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
