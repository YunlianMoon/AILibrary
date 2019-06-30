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
- policy based

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
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/policy_gradient_1.png" width="40%" /><br/>
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
  Policy Gradient
</div>

### Value-based Approach (Learning a Critic)
<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/critic_1.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/critic_2.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/critic_3.png" width="30%" />
  <br/>
  Three kinds of Critics
</div>

<br/>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/V_MC.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/V_TD.png" width="40%" />
  <br/>
  How to estimate 𝑉(s)
</div>

### Actor-Critic
<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/actor_critic_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/actor_critic_2.png" width="40%" />
  <br/>
  Actor-Critic
</div>

### Asynchronous Advantage Actor-Critic (A3C)
Asynchronous methods for deep reinforcement learning \[2016, ICML, Volodymyr Mnih\] \[[paper](http://www.jmlr.org/proceedings/papers/v48/mniha16.pdf)\]

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/A3C.png" width="40%" />
  <br/>
  A3C
</div>

### Proximal Policy Optimization (PPO)
default reinforcement learning algorithm at OpenAI

demo: \[[DeepMind](https://www.youtube.com/watch?v=gn4nRCC9TwQ&feature=youtu.be)\] \[[OpenAI]()\]





