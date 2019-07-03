# Sparse Reward

- Reference
  - [Machine Learning and having it deep and structured (2018,Spring)](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)
  
### Reward Shaping

Training agent for first-person shooter game with actor-critic curriculum learning \[2016, Yuxin Wu\] \[[paper](https://openreview.net/pdf?id=Hk3mPK5gg)\]

#### Curiosity

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/curiosity_1.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/curiosity_2.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/curiosity_3.png" width="30%" /><br/>
  Intrinsic Curiosity Module
</div>

Curiosity-driven exploration by self-supervised prediction \[2017, CVPR, Deepak Pathak\] \[[paper](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/papers/Pathak_Curiosity-Driven_Exploration_by_CVPR_2017_paper.pdf)\]

#### Reward from Auxiliary Task 

Reinforcement learning with unsupervised auxiliary tasks \[2016, arxiv, Max Jaderberg\] \[[paper](https://arxiv.org/pdf/1611.05397.pdf)\]

### Curriculum Learning

Starting from simple training examples, and then becoming harder and harder

#### Reverse Curriculum Generation

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/reverse_curriculum_1.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/ReinforcementLearning/images/reverse_curriculum_2.png" /><br/>
  Reverse Curriculum Generation
</div>

### Hierarchical Reinforcement Learning

- If lower agent cannot achieve the goal, the upper agent would get penalty
- If an agent get to the wrong goal, assume the original goal is the wrong one

Hierarchical reinforcement learning with hindsight \[2018, arxiv, Andrew Levy\] \[[paper]()\]



