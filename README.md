# Non-ergodic Deep Reinforcement Learning

PyTorch implementation investigating the impact of different loss functions for critic backpropagation and (eventually) implementing non-ergodicity.
Focus is on existing state-of-the-art model-free algorithms TD3 and SAC using OpenAI gym and PyBullet environments.

Agents are trained on a diverse range of environments. Form OpenAI gym: 'LunarLanderContinuous-v2', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3'. From PyBullet:  'CartPoleContinuousBulletEnv-v0', 'InvertedPendulumBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0', 'HumanoidBulletEnv-v0'.

Loss functions used include MSE, MAE, Huber, Hypersurface Cost, Cauchy and Truncated Cauchy. Scale Parameter for Cauchy distribution is estimated using the Nagy algorithm and truncation is performed using heuristics. Non-ergodicity model is currently being tested.

This repository will be submitted by mid-2021 as a component of ‘DATA5709: Capstone Project – Individual’ in partial fulfilment of the requirements of the Masters of Data Science at the University of Sydney, Australia. 

Code tested both locally on an AMD Ryzen 7 5800X, Nvidia RTX 3070, 64GB 3200MHz CL16 RAM, Samsung 980 Pro and on the Artemis HPC at the University of Sydney.

## Algorithms Utilised
* Twin Delayed Deep Deterministic Policy Gradients (TD3) ([Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477))
* Soft Actor-Critic (SAC) ([Haarnoja et al. 2018a](https://arxiv.org/pdf/1801.01290.pdf), [Haarnoja et al. 2018b](https://arxiv.org/pdf/1812.05905.pdf), [Haarnoja et al. 2019](https://arxiv.org/pdf/1812.11103.pdf))

## Cauchy Function Scale Estimation and Truncation
* Nagy Algorithm ([Nagy 2006](http://www.jucs.org/jucs_12_9/parameter_estimation_of_the/jucs_12_09_1332_1344_nagy.pdf))
* Truncated Cauchy ([Guan et al. 2019](https://tongliang-liu.github.io/papers/TPAMITruncatedNMF.pdf))


## Comments on Implementation
TD3 appears to be well-optimised and functioning. Ready to parallelise across multiple GPUs to run 10 trials, each of 1e6 episodes per environment per loss function.

SAC currently requires tuning to transfer high CPU usage (8 threads at 100% utilisation) to the GPU. Algorithm also appears for certain environments to struggle in learning multi-modal solutions. Additionally, our use of true multi-dimensional stochastic Gaussian noise added to each policy action component inside every sample contained the mini-batch appears to reduce performance compared the simpler Gaussian sampling used by the overwhelming majority of available implementations.

## Potential Applications
Existing portfolio management tool in finance, systems control, risk management systems all to some degree rely on the use of expectation values (probability weighed averages) which are inherently and deeply flawed since no individual or institution ever experiences the ‘expected’ value. 

This is because extremely important behaviour occurs during the pre-asymptotic phase and so the law of large numbers or Monte Carlo approach is flawed ([Taleb 2020]( https://arxiv.org/ftp/arxiv/papers/2001/2001.10488.pdf)). Furthermore the use of probabilities is questionable since values below 0 and above 1 not only exist, but have real physical interpretations ([Gell-Mann and Hartle 2011]( https://arxiv.org/pdf/1106.0767.pdf)). Finally the majority of classical decision theory is built upon an incorrect 300 year old assumption that maximising utility requires the use bounded utility functions ([Peters and Gell-Mann 2015]( https://arxiv.org/pdf/1405.0585.pdf)).

## Acknowledgements
The author acknowledge the facilities, and the scientific and technical assistance of the Sydney Informatics Hub at the University of Sydney and, in particular, access to the high performance computing facility Artemis.

The python implementation has been significantly modified but is based on the original authors’ code along with insight from several other repositories. Below is an alphabetised list of sources.
* [p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch]( https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
* [philtabor/Actor-Critic-Methods-Paper-To-Code](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code)
* [rail-berkley/softlearning]( https://github.com/rail-berkeley/softlearning) 
* [rlworkgroup/garage](https://github.com/rlworkgroup/garage)
* [sfujim/TD3](https://github.com/sfujim/TD3/)
