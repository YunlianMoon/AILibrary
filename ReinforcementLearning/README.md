# Deep Reinforcement Learning

### Table of Contents
- value-based
  - <a href="#DQN">DQN</a>
  - <a href="#DoubleDQN">Double DQN</a>
  - <a href="#PrioritizedExperienceReplayDQN">Prioritized Experience Replay DQN</a>
  - <a href="#DuelingDQN">Dueling DQN</a>
- policy gradient
  - <a href="#REINFORCE">REINFORCE</a>
- actor critic
  - <a href="#ActorCritic">Actor Critic</a>

#### <a name="DQN">DQN</a>
paper: [Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).](https://arxiv.org/pdf/1312.5602.pdf\))

![DQN](https://latex.codecogs.com/svg.latex?R&plus;%5Cgamma%20%5Cunderset%7Ba%7D%7Bmax%7D%20Q%28S_%7Bt&plus;1%7D%2Ca%3B%5Ctheta_%7Bt%7D%5E%7B-%7D%20%29 "DQN")

#### <a name="DoubleDQN">Double DQN</a>
paper: [Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." Thirtieth AAAI Conference on Artificial Intelligence. 2016.](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847)

![DoubleDQN](https://latex.codecogs.com/svg.latex?R&plus;%5Cgamma%20Q%28S_%7Bt&plus;1%7D%2C%5Cunderset%7Ba%7D%7Bargmax%7DQ%28S_%7Bt&plus;1%7D%2Ca%3B%5Ctheta_%7Bt%7D%29%3B%5Ctheta_%7Bt%7D%5E%7B-%7D%20%29 "DoubleDQN")

#### <a name="PrioritizedExperienceReplayDQN">Prioritized Experience Replay DQN</a>
paper: [Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).](https://arxiv.org/pdf/1511.05952.pdf)

#### <a name="DuelingDQN">Dueling DQN</a>
paper: [Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." arXiv preprint arXiv:1511.06581 (2015).](https://arxiv.org/pdf/1511.06581.pdf)

![DuelingDQN](https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/DuelingDQN.svg "DuelingDQN")

#### <a name="REINFORCE">REINFORCE</a>
paper: [Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with function approximation." Advances in neural information processing systems. 2000.](http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

![REINFORCE](https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/REINFORCE.svg "REINFORCE")



