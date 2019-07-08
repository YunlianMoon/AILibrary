# Anomaly Detection

- Reference
  - [Machine Learning (2019,Spring)](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html)
- Applications
  - Fraud Detection
    - Training data: 正常刷卡行為, 𝑥: 盜刷？ \[[link](https://www.kaggle.com/ntnu-testimon/paysim1/home)\] \[[link](https://www.kaggle.com/mlg-ulb/creditcardfraud/home)\]
  - Network Intrusion Detection
    - Training data: 正常連線, 𝑥: 攻擊行為？ \[[link](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)\]
  - Cancer Detection
    - Training data: 正常細胞, 𝑥: 癌細胞？ \[[link](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/home)\]
  
<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/anomaly_detection.png" width="40%" /><br/>
  Problem Formulation
</div>

<br/>
    
<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/anomaly_detection_category_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/anomaly_detection_category_2.png" width="40%" /><br/>
  Categories
</div>

### With Classifier

Simpsons dataset \[[link](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset/)\]

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/anomaly_example_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/anomaly_example_2.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/anomaly_example_3.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/anomaly_example_4.png" width="50%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/anomaly_example_5.png" width="40%" /><br/>
  Example Application
</div>

Learning confidence for out-of-distribution detection in neural networks \[2018, arxiv, Terrance DeVries\] \[[paper](https://arxiv.org/pdf/1802.04865.pdf)\]

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/anomaly_evaluation_1.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/anomaly_evaluation_2.png" width="40%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/anomaly_evaluation_3.png" width="40%" /><br/>
  Evaluation
</div>

<br/>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/issue_1.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/issue_2.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/issue_3.png" width="30%" /><br/>
  Possible Issues
</div>

Novelty detection with gan \[2018, arxiv, Mark Kliger\] \[[paper](https://arxiv.org/pdf/1802.10560.pdf)\]

Training confidence-calibrated classifiers for detecting out-of-distribution samples \[2018, arxiv, Kimin Lee\] \[[paper](https://arxiv.org/pdf/1711.09325.pdf)\]

### Without Labels

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/no_label_1.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/no_label_2.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/no_label_3_3.png" width="30%" /><br/>
  Problem Formulation
</div>

twitch-troll-detection \[[link](https://github.com/ahaque/twitch-troll-detection)\]

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/maximum_likelihood_1.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/maximum_likelihood_2.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/maximum_likelihood_3.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/maximum_likelihood_4.png" width="30%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/DeepLearning/Attention/images/arrow.jpg" width="2%" />
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/maximum_likelihood_5.png" width="30%" /><br/>
  Maximum Likelihood
</div>

<br/>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/auto-encoder.png" width="30%" /><br/>
  Auto-encoder
</div>

<br/>

<div align=center>
  <img src="https://github.com/YunlianMoon/AILibrary/blob/master/next_machine_learning/images/more.png" width="30%" /><br/>
  More
</div>


