# Deep Reinforcement Learning

- Reference
  - [Machine Learning (2016,Fall)](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML16.html)
  - [Machine Learning and having it deep and structured (2018,Spring)](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)
  - [Machine Learning (2019,Spring)](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html)
  
### Application
- chat-bot <br/>
Deep reinforcement learning for dialogue generation \[2016, arxiv, Jiwei Li\] \[[paper](https://arxiv.org/pdf/1606.01541v3.pdf)\]
- Interactive retrieval
- Flying Helicopter
- Driving
- Google Cuts Its Giant Electricity Bill With DeepMindPowered AI
- Text generation <br/>
Generating Text with Deep Reinforcement Learning \[2015, NIPS, Hongyu Guo\] \[[paper](https://arxiv.org/pdf/1510.09202.pdf)\] <br/>
Sequence level training with recurrent neural networks \[2015, arxiv, Marc'Aurelio Ranzato\] \[[paper](https://arxiv.org/pdf/1511.06732.pdf)\]
- Playing Video Game
  - Space invader
- paly Go

### Difficulties of Reinforcement Learning
- Reward delay
- Agent’s actions affect the subsequent data it receives

### Classification
- value based
  - Q learning
  - DQN/Double DQN/Dueling DQN/Prioritized Reply/Multi-step/Noisy Net/Distributional Q-function/Rainbow
  - Q-Learning for Continuous Actions
- policy based
  - policy gradient
  - TRPO
  - PPO
- Actor-Critic
  - A2C
  - A3C
  - Pathwise Derivative Policy Gradient

### Policy-based Approach (Learning an Actor)
<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/NN_actor.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/actor_goodness_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/actor_goodness_2.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/gradient_ascent_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/gradient_ascent_2.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/gradient_ascent_3.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/gradient_ascent_4.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/gradient_ascent_5.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/gradient_ascent_6.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/baseline.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/advantage.png" width="40%" />
  <br/>
  Policy-based Approach
</div>

<br/>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/policy_gradient_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/policy_gradient_2.png" width="40%" /><br/>
  Policy Gradient
</div>

### On-policy v.s. Off-policy
- On-policy: The agent learned and the agent interacting with the environment is the same.
- Off-policy: The agent learned and the agent interacting with the environment is different.

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/IS_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/IS_issue_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/IS_issue_2.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/IS_2.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/IS_3.png" width="40%" /><br/>
  On-policy to Off-policy
</div>

### Proximal Policy Optimization (PPO)
default reinforcement learning algorithm at OpenAI

demo: \[[DeepMind](https://www.youtube.com/watch?v=gn4nRCC9TwQ&feature=youtu.be)\] \[[OpenAI]()\]

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/ppo_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/ppo_2.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/ppo_3.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/ppo_4.png" width="40%" /><br/>
  PPO/TRPO
</div>

Proximal policy optimization algorithms \[2017, arxiv, John Schulman\] \[[paper](https://arxiv.org/pdf/1707.06347.pdf)\]

### Value-based Approach (Learning a Critic)
<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/critic_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/critic_2.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/critic_3.png" width="40%" />
  <br/>
  Two kinds of Critics
</div>

<br/>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/V_MC.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/V_TD.png" width="40%" />
  <br/>
  How to estimate 𝑉(s) (Estimated by TD or MC)
</div>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/MC_TD_compare.png" width="40%" /><br/>
  MC v.s. TD
</div>

### Q-Learning
<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/q_learning_1.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/q_learning_2.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/q_learning_3.png" width="40%" /><br/>
  Q-Learning
</div>

<br/>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/q_learning_tips_target_network.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/q_learning_tips_exploration.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/q_learning_tips_replay_buffer_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/q_learning_tips_replay_buffer_2.png" width="40%" /><br/>
  Q-Learning technologies
</div>

<br/>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/typical_q_learning_algorithm.png" width="40%" /><br/>
  Typical Q-Learning Algorithm
</div>

#### Tips of Q-Learning

- Double DQN

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/double_dqn_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/double_dqn_2.png" width="40%" /><br/>
  Double DQN
</div>

Deep reinforcement learning with double q-learning \[2016, AAAI, Hasselt Van\] \[[paper](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Applications_files/doubledqn.pdf)\]

- Dueling DQN

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/dueling_dqn_1.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/dueling_dqn_2.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/dueling_dqn_3.png" width="30%" /><br/>
  Dueling DQN
</div>

Dueling network architectures for deep reinforcement learning \[2015, arxiv, Ziyu Wang\] \[[paper](https://arxiv.org/pdf/1511.06581.pdf)\]

- Prioritized Reply

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/prioritized_reply.png" width="40%" /><br/>
  Prioritized Reply
</div>

Prioritized experience replay \[2015, arxiv, Tom Schaul\] \[[paper](https://arxiv.org/pdf/1511.05952.pdf)\]

- Multi-step

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/multi_step.png" width="40%" /><br/>
  Multi-step
</div>

- Noisy Net

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/noise_net_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/noise_net_2.png" width="40%" /><br/>
  Noisy Net
</div>

Parameter space noise for exploration \[2017, arxiv, Matthias Plappert\] \[[paper](https://arxiv.org/pdf/1706.01905.pdf)\]

Noisy networks for exploration \[2017, arxiv, Meire Fortunato\] \[[paper](https://arxiv.org/pdf/1706.10295.pdf)\]

demo: \[[link](https://openai.com/blog/better-exploration-with-parameter-noise/)\]

- Distributional Q-function

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/distributional_q_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/distributional_q_2.png" width="40%" /><br/>
  Distributional Q-function
</div>

demo: \[[link](https://www.youtube.com/watch?v=yFBwyPuO2Vg&feature=youtu.be)\]

- Rainbow

Rainbow: Combining improvements in deep reinforcement learning \[2018, AAAI, Matteo Hessel\] \[[paper](https://arxiv.org/pdf/1710.02298.pdf)\]

#### Q-Learning for Continuous Actions

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/continuous_action_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/continuous_action_2.png" width="40%" /><br/>
  Q-Learning for Continuous Actions
</div>

demo: \[[link](https://www.youtube.com/watch?v=ZhsEKTo7V04)\]

### Actor-Critic
<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/actor_critic_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/actor_critic_2.png" width="40%" /><br/>
  Actor-Critic
</div>

#### Advantage Actor-Critic (A2C)

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/advantage_actor_critic_1.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/advantage_actor_critic_2.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/advantage_actor_critic_3.png" width="30%" /><br/>
  Advantage Actor-Critic
</div>

#### Asynchronous Advantage Actor-Critic (A3C)
Asynchronous methods for deep reinforcement learning \[2016, ICML, Volodymyr Mnih\] \[[paper](http://www.jmlr.org/proceedings/papers/v48/mniha16.pdf)\]

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/A3C.png" width="40%" />
  <br/>
  A3C
</div>

#### Pathwise Derivative Policy Gradient

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/pathwise_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/pathwise_2.png" width="40%" /><br/>
  Pathwise Derivative Policy Gradient
</div>

<br/>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/q_learning_algorithm.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/pathwise_algorithm.png" width="40%" /><br/>
  Pathwise Derivative Policy Gradient algorithm
</div>

<br/>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/ac_gan.png" width="40%" /><br/>
  Connection with GAN
</div>

Deterministic policy gradient algorithms \[2014, ICML, David Silver\] \[[paper](http://proceedings.mlr.press/v32/silver14.pdf)\]

Continuous control with deep reinforcement learning \[2015, arxiv, Timothy P. Lillicrap\] \[[paper](https://arxiv.org/pdf/1509.02971.pdf)\]

Connecting generative adversarial networks and actor-critic methods \[2016, arxiv, David Pfau\] \[[paper](https://arxiv.org/pdf/1610.01945.pdf)\]





