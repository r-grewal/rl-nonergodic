# Non-ergodic Deep Reinforcement Learning

PyTorch implementation investigating the impact of different loss functions for critic backpropagation and (eventually) implementing non-ergodicity.
Focus is on existing state-of-the-art model-free algorithms TD3 and SAC using OpenAI gym and PyBullet environments.

Loss functions used include MSE, MAE, Huber, Hypersurface Cost, Cauchy and Truncated Cauchy. Scale Parameter for Cauchy distribution is estimated using the Nagy algorithm and truncation is performed using heuristics.

This project will be submitted by mid-2021 as part of the requirements of ‘DATA5709 Capstone Project – Individual’ at the University of Sydney, Australia. Code tested localy on AMD Ryzen 7 5800X, Nvidia RTX 3070, 64GB 3200MHz RAM and on Artemis HPC.

## Algorithms Utilised
* Twin Delayed Deep Deterministic Policy Gradients (TD3) ([Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477))
* Soft Actor-Critic (SAC) ([Haarnoja et al. 2018](https://arxiv.org/pdf/1801.01290.pdf), [Haarnoja et al. 2018](https://arxiv.org/pdf/1812.05905.pdf), [Haarnoja et al. 2019](https://arxiv.org/pdf/1812.11103.pdf))

## Cauchy Function Scale Estimation and Truncation
* Nagy Algorithm ([Nagy 2006](http://www.jucs.org/jucs_12_9/parameter_estimation_of_the/jucs_12_09_1332_1344_nagy.pdf))
* Truncated Cauchy ([Guan et al. 2019](https://tongliang-liu.github.io/papers/TPAMITruncatedNMF.pdf))

## Potential Applications
Existing portfolio management tool in finance, systems control, risk management systems all to some degree rely on the use of expectation values (probability weighed averages) which are inherently and deeply flawed since no individual or institution ever experiences the ‘expected’ value. 

## Acknowledgements
The author acknowledge the facilities, and the scientific and technical assistance of the Sydney Informatics Hub at the University of Sydney and, in particular, access to the high performance computing facility Artemis.

The python implementation has been significantly modified but is based on the original authors’ code along with insight from several other repositories. Below is a alphabetised list of sources.
* [p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch]( https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
* [philtabor/Actor-Critic-Methods-Paper-To-Code](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code)
* [rail-berkley/softlearning]( https://github.com/rail-berkeley/softlearning) 
* [rlworkgroup/garage](https://github.com/rlworkgroup/garage)
* [sfujim/TD3](https://github.com/sfujim/TD3/)



