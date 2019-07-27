# Code
import tensorflow as tf

### RNN
lstm_cell = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)<br/>
lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

### Loss
loss = tf.reduce_mean(tf.square(pred - targets), name="mse")<br/>
loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)), name="cross_entropy")

softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=pred)<br/>
softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=pred)<br/>
softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=pred)<br/>
sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=pred)<br/>
weighted_cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets, logits, pos_weight)

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


### Optimizer
optim = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, name="rmsprop_optim")
