# Code
import tensorflow as tf

### RNN
lstm_cell = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)<br/>
lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

### Loss
loss = tf.reduce_mean(tf.square(pred - targets), name="mse")<br/>
loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)), name="cross_entropy")

### Optimizer
optim = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, name="rmsprop_optim")
