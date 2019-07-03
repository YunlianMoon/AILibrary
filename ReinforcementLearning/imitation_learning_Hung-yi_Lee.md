# Imitation Learning

- Reference
  - [Machine Learning and having it deep and structured (2018,Spring)](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)
  
- Imitation Learning
  - Also known as learning by demonstration, apprenticeship learning
- An expert demonstrates how to solve the task
  - Machine can also interact with the environment, but cannot explicitly obtain reward
  - It is hard to define reward in some tasks
  - Hand-crafted rewards can lead to uncontrolled behavior
- Two approaches
  - Behavior Cloning
  - Inverse Reinforcement Learning (inverse optimal control)

### Behavior Cloning

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/behavior_cloning.png" width="40%" /><br/>
  Behavior Cloning
</div>

#### problem

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/behavior_problem_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/behavior_problem_2.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/behavior_problem_3.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/behavior_problem_4.png" width="40%" />
  <br/>
  Behavior Cloning Problem
</div>

The agent will copy every behavior, even irrelevant actions \[[video](https://www.youtube.com/watch?v=j2FSB3bseek)\]

### Inverse Reinforcement Learning (IRL)

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/inverse_rl_1.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/inverse_rl_2.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/inverse_rl_3.png" width="30%" />
  <br/>
  Inverse Reinforcement Learning
</div>

Guided cost learning: Deep inverse optimal control via policy optimization \[2016, Chelsea Finn, ICML\] \[[paper](http://proceedings.mlr.press/v48/finn16.pdf)\]

### Third Person Imitation Learning

Third-person imitation learning \[2017, arxiv, Bradly C. Stadie\] \[[paper](https://arxiv.org/pdf/1703.01703.pdf)\]


