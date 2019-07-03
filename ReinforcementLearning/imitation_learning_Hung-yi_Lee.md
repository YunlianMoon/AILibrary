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


