# Sequence Generation

- Reference
  - Machine Learning (2019,Spring) \[[link](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html)\] \[[video](https://www.youtube.com/watch?v=ZjfjPzXw6og&feature=youtu.be)\] \[[pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/Seq%20(v2).pdf)\]

### RNN with Gated Mechanism

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/RNN.png" width="40%" /><br/>
  Recurrent Neural Network<br>
  (No matter how long the input/output sequence is, we only need one function f)
</div>

<br/>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/Deep_RNN.png" width="40%" /><br/>
  Deep Recurrent Neural Network
</div>

<br/>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/Bidirectional_RNN.png" width="40%" /><br/>
  Bidirectional Recurrent Neural Network
</div>

#### Naive RNN

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/naive_RNN.png" width="40%" /><br/>
  Naïve Recurrent Neural Network
</div>

#### LSTM

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/LSTM.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/LSTM_1.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/LSTM_2.png" width="30%" />
  <br/>
  Long Short Term Memory
</div>

<br>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/LSTM_peephole.png" width="40%" /><br/>
  peephole
</div>

LSTM: A search space odyssey \[2016, arxiv, Klaus Greff\] \[[paper](https://arxiv.org/pdf/1503.04069.pdf?utm_content=buffereddc5&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer)\]

#### GRU

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/GRU.png" width="40%" /><br/>
  GRU
</div>

### Sequence Generation

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/generation.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/generation_1.png" width="40%" /><br/>
  Sequence Genration
</div>

### Conditional Sequence Generation

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/image_caption.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/translation.png" width="40%" /><br/>
  Conditional Generation
</div>

Building end-to-end dialogue systems using generative hierarchical neural network models \[2016, AAAI, Iulian V. Serban\] \[[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11957/12160)\]

### Tips for Generation

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/scheduled_sampling.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/RNN/images/beam_search.png" width="40%" /><br/>
  Testing: The inputs are the outputs of the last time step. Training: The inputs are reference.
</div>



