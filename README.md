# Robust Twin Critic Evaluation, n-step Returns and Non-ergodicity in Deep Reinforcement Learning

PyTorch implementation investigating the impact of different surrogate loss functions for critic backpropagation, n-step returns in continuous action space and (eventually) implementing non-ergodicity.
Focus is on existing state-of-the-art, off-policy, model-free algorithms TD3 and SAC using PyBullet environments with additional algorithms added soon.

Agents are trained on a diverse range of popular RL environments from 3D hopping to full upright 3D humanoid movement. Below we list the four to be initially tested with input features and continuous actions dimensions showed in brackets respectively.
* HopperBulletEnv-v0 (15, 3), Walker2DBulletEnv-v0 (22, 6), AntBulletEnv-v0 (28, 8), HumanoidBulletEnv-v0 (44, 17).
Some comparative results are presented in [Raffin and Stulp (2020)](https://arxiv.org/pdf/2005.05719.pdf).

Loss functions used in descending order outlier suppression include even powers of MSE up to MSE^4, MSE, Huber, MAE, Hypersurface Cost, Cauchy, Truncated Cauchy and Correntropy-Induced Metric. Scale Parameter for Cauchy distribution is estimated using the Nagy algorithm and truncation is performed using heuristics. Correntropy Gaussian kernel size is estimated empirically as the average error. Non-ergodicity model is currently being tested in suitable environments. Furthermore, experiments will be conducted on varying the size of the replay buffer and incorporating n-step returns in continuous action spaces for the critic network.

This repository will be submitted on 20/06/2021, the end of my final semester, as the key component of ‘DATA5709: Capstone Project – Individual’ in partial fulfilment of the requirements of the Master of Data Science at the University of Sydney. 

Code tested locally on both Windows 10 20H2 and Ubuntu 20.04.02 LTS using an AMD Ryzen 7 5800X (5.1GHz), Nvidia RTX 3070, 64GB 3600MHz CL16 RAM and Samsung 980 Pro. The final results will be gathered on the CentOS 6.9 Linux-based Artemis HPC.

See progress_log.md for detailed update history.

## References
* Deep Deterministic Policy Gradients (DDPG) ([Silver et al. 2014](http://proceedings.mlr.press/v32/silver14.pdf), [Lillicrap et al. 2016](https://arxiv.org/pdf/1509.02971.pdf))
* Proximal Policy Optmisation (PPO) ([Schulmanet al. 2017](https://arxiv.org/pdf/1707.06347.pdf))
* Twin Delayed Deep Deterministic Policy Gradients (TD3) ([Fujimoto et al. 2018](https://arxiv.org/pdf/1802.09477.pdf))
* Soft Actor-Critic (SAC) ([Haarnoja et al. 2017](https://arxiv.org/pdf/1702.08165.pdf), [Haarnoja et al. 2017](https://arxiv.org/pdf/1702.08165.pdf), [Haarnoja et al. 2018a](https://arxiv.org/pdf/1801.01290.pdf), [Haarnoja et al. 2018b](https://arxiv.org/pdf/1803.06773.pdf))
* Soft Actor-Critic (SAC) w/ Learned Temperature ([Haarnoja et al. 2018c](https://arxiv.org/pdf/1812.05905.pdf), [Haarnoja et al. 2019](https://arxiv.org/pdf/1812.11103.pdf))
* Augmented Random Search (ARS) ([Mania, Guy and Recht 2018]( https://arxiv.org/pdf/1803.07055.pdf)) 
* Coupling of n-step Returns and Experience Replay ([Fedus et al. 2020](https://arxiv.org/pdf/2007.06700.pdf))
* Hypersurface Loss ([Samson et al. 2000](https://ieeexplore.ieee.org/document/857003), [Hamza and Brady 2006](https://users.encs.concordia.ca/~hamza/HamzaBrady.pdf))
* Correntropy and Huber Loss ([Lium, Pokharel, Principe 2007]( https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=22F68E41A9F19D343CCF47403C29038F?doi=10.1.1.640.6891&rep=rep1&type=pdf), [Pokharel and Principe 2012](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.639.4052&rep=rep1&type=pdf), [Du, Li and Shen 2012](https://lcs.ios.ac.cn/~ydshen/ICDM-12.pdf))
* Cauchy Scale Estimation via the Nagy Algorithm ([Nagy 2006](http://www.jucs.org/jucs_12_9/parameter_estimation_of_the/jucs_12_09_1332_1344_nagy.pdf))
* Truncated Cauchy Loss ([Guan et al. 2019](https://tongliang-liu.github.io/papers/TPAMITruncatedNMF.pdf))

## Potential Applications
Existing systems control, portfolio management tool in finance, risk management systems all to some degree rely on the use of continuous expectation values (probability weighted averages) which are inherently and deeply flawed since literally no individual or institution ever experiences the ‘expected’ value. 

Given extremely important behaviour and events are very likely to occur during the pre-asymptotic phase, the use of the law of large numbers or Monte Carlo approach is not valid in realistic and mission-critical scenarios where there is either minimal available data or where extreme losses are irrecoverable ([Taleb 2020]( https://arxiv.org/ftp/arxiv/papers/2001/2001.10488.pdf)). Furthermore the use of probabilities is questionable since values below 0 and above 1 not only exist, but have real physical interpretations ([Gell-Mann and Hartle 2011]( https://arxiv.org/pdf/1106.0767.pdf)). Finally the majority of classical decision theory, economics and behavioural finance is built upon an 300 year old error that assumes maximising utility requires the use bounded utility functions which leads to serious underpricing pricing of risk ([Peters and Gell-Mann 2015]( https://arxiv.org/pdf/1405.0585.pdf)).

## Acknowledgements
The author acknowledges the facilities, and the scientific and technical assistance of the Sydney Informatics Hub at the University of Sydney and, in particular, access to the high performance computing facility Artemis.

The python implementation of base algorithms has been significantly modified and written from scratch but is based on the original authors’ code along with insight from several other repositories. Below is an alphabetised list of sources.
* [DLR-RM/stable-baelines3](https://github.com/DLR-RM/stable-baselines3)
* [haarnoja/sac](https://github.com/haarnoja/sac)
* [openai/spinningup](https://github.com/openai/spinningup)
* [p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch]( https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
* [philtabor/Actor-Critic-Methods-Paper-To-Code](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code)
* [rail-berkley/softlearning]( https://github.com/rail-berkeley/softlearning) 
* [rlworkgroup/garage](https://github.com/rlworkgroup/garage)
* [sfujim/TD3](https://github.com/sfujim/TD3/)
