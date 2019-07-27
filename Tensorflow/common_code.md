# Code
import tensorflow as tf

### RNN
---
lstm_cell = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)<br/>
lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

### Loss
---
#### regression
(1)<br/>
mse_loss = tf.reduce_mean(tf.square(pred - targets))<br/>
mse_loss = tf.losses.mean_squared_error(targets, pred)

(2)<br/>
maes_loss = tf.reduce_sum(tf.losses.absolute_difference(targets, pred))

(3)<br/>
hubers_loss = tf.reduce_sum(tf.losses.huber_loss(targets, pred))

#### classification
(1)<br/>
softmax_cross_entropy = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))<br/>
softmax_cross_entropy_loss = tf.reduce_mean(cross_entropy)

softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=pred)<br/>
softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=pred)<br/>
softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=pred)
``` python
import tensorflow as tf

logits = [[2, 0.5, 1],
          [0.1, 1, 3]]
labels = [[0.2, 0.3, 0.5],
          [0.1, 0.6, 0.3]]

logits_scaled = tf.nn.softmax(logits)

result1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) # [1.4143689 1.6642545]
result2 = -tf.reduce_sum(labels * tf.log(logits_scaled), 1) # [1.4143689 1.6642545]

labels1 = [[0, 1, 0],
          [0, 0, 1]]
labels2 = [1, 2]

result3 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels1) # [1.9643688  0.17425454]
result4 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels2) # [1.9643688  0.17425454]
```

(2)<br/>
y_pred_si = 1.0 / (1 + tf.exp(-pred))<br/>
sigmoids_cross_entropy = -targets * tf.log(y_pred_si) - (1 - targets) * tf.log(1 - y_pred_si)<br/>
sigmoids_cross_entropy_loss = tf.reduce_mean(sigmoids)

sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=pred)<br/>
sigmoids_cross_entropy = tf.losses.log_loss(labels=targets, logits=pred)<br/>
weighted_cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets, logits, pos_weight)
``` python
import numpy as np
import tensorflow as tf

labels = np.array([[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]])
logits = np.array([[-800, -2., -3],
                   [-10., -700., -13.],
                   [-1., -2., 0]])

y_pred = 1.0 / (1 + np.exp(-logits))
result1 = -labels * np.log(y_pred) - (1 - labels) * np.log(1 - y_pred)
result2 = logits - logits * labels + np.log(1 + np.exp(-logits))
'''
result1=result2=
[[           inf 1.26928011e-01 4.85873516e-02]
 [4.53988992e-05 7.00000000e+02 2.26032685e-06]
 [3.13261688e-01 1.26928011e-01 6.93147181e-01]]
'''

result3 = np.greater_equal(logits, 0) - logits * labels + np.log(1 + np.exp(-np.abs(logits)))
result4 = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
'''
result3=result4=
[[8.00000000e+02 1.26928011e-01 4.85873516e-02]
 [4.53988992e-05 7.00000000e+02 2.26032685e-06]
 [3.13261688e-01 1.26928011e-01 1.69314718e+00]]
'''
```

(3)<br/>
hings = tf.losses.hinge_loss(labels=targets, logits=pred, weights)
hings_loss = tf.reduce_mean(hings)



### Optimizer
---
optim = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, name="rmsprop_optim")
