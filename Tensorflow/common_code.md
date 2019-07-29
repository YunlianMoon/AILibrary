# Code
import tensorflow as tf

### RNN
---
(1) cell<br/>
gru_cell = tf.nn.rnn_cell.GRUCell(num_units=)<br/>
gru_cell = tf.contrib.rnn.GRUCell(num_units=)

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=, forget_bias=, state_is_tuple=)<br/>
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=, forget_bias=, state_is_tuple=)<br/>
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=, forget_bias=, state_is_tuple=)<br/>
lstm_cell = tf.contrib.rnn.LSTMCell(num_units=, forget_bias=, state_is_tuple=)

multi_cell= tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=)<br/>
multi_cell= tf.contrib.rnn..MultiRNNCell(cells, state_is_tuple=)

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=)<br/>
rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=)<br/>
rnn_cell = tf.nn.rnn_cell.RNNCell()<br/>
rnn_cell = tf.contrib.rnn.RNNCell()

(2) dropout wrapper<br/>
cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=, output_keep_prob=)<br/>
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=, output_keep_prob=)<br/>

(3)<br/>
lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
    cell,
    inputs,
    sequence_length=None,
    initial_state=None,
    dtype=None,
    parallel_iterations=None,
    swap_memory=False,
    time_major=False,
    scope=None
)

lstm_outputs, lstm_state = tf.nn.bidirectional_dynamic_rnn(
    cell_fw,
    cell_bw,
    inputs,
    sequence_length=None,
    initial_state_fw=None,
    initial_state_bw=None,
    dtype=None,
    parallel_iterations=None,
    swap_memory=False,
    time_major=False,
    scope=None
)
``` python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist dataset: http://yann.lecun.com/exdb/mnist/

# hyper parameters
LEARNING_RATE = 0.01  # learning rate
BATCH_SIZE = 128
INPUT_SIZE = 28  # MNIST data input (img shape: 28*28)
STEP_SIZE = 28  # time steps
NUM_CLASSES = 10

# load mnist data
data_dir = 'MNIST_data'
mnist = input_data.read_data_sets(data_dir, one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# x y placeholder
x = tf.placeholder(tf.float32, [None, STEP_SIZE * INPUT_SIZE])  # shape(batch, 784)
image = tf.reshape(x, [-1, STEP_SIZE, INPUT_SIZE])  # (batch, height, width, channel)
y = tf.placeholder(tf.int32, [None, NUM_CLASSES])  # input y

# RNN
# cell = tf.nn.rnn_cell.BasicRNNCell(num_units=64) # 0.72
# cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64) # 0.91
# cell = tf.nn.rnn_cell.LSTMCell(num_units=64) # 0.92
cell = tf.nn.rnn_cell.GRUCell(num_units=64)  # 0.93
outputs, final_state = tf.nn.dynamic_rnn(
    cell,  # cell you have chosen
    image,  # input
    initial_state=None,  # the initial hidden state
    dtype=tf.float32,  # must given if set initial_state = None
    time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
)
# outputs:[batch，time step，n_neurons]

output = tf.layers.dense(outputs[:, -1, :], NUM_CLASSES)  # output based on the last output step

loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)  # compute cost
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(output, axis=1), )[1]
# return (acc, update_op), and create 2 local variables

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())  # the local var is for accuracy_op
sess.run(init_op)  # initialize var in graph

for step in range(1200):  # training
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {x: b_x, y: b_y})
    if step % 50 == 0:  # testing
        accuracy_ = sess.run(accuracy, {x: test_x, y: test_y})
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
```

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
softmax_cross_entropy_loss = tf.reduce_mean(softmax_cross_entropy)<br/>
loss = tf.losses.softmax_cross_entropy(onehot_labels=targets, logits=pred)

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
sigmoids_cross_entropy_loss = tf.reduce_mean(sigmoids_cross_entropy)

sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=pred)<br/>
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
hings = tf.losses.hinge_loss(labels=targets, logits=pred, weights)<br/>
hings_loss = tf.reduce_mean(hings)

### Optimizer
---
optim = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, name="rmsprop_optim")
