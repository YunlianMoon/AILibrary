# TensorFlow

import tensorflow as tf

### Basic Usage

#### source op
- tf.placeholder(dtype, shape=None, name=None)
- tf.Variable(initializer, name)

- tf.zeros(shape, dtype=tf.float32, name=None)
- tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
- tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)


#### session
init = tf.initialize_all_variables()

(1) <br />
sess = tf.Session() <br />
sess.run(init) <br />
fetch: <br />
sess.run() <br />
(2) <br />
sess = tf.InteractiveSession() <br />
sess.run(init) <br />
fetch: <br />
Tensor.eval() <br />
Operation.run() <br />
(3) <br />
with tf.Session() as sess: <br />
  sess.run(init) <br />

sess.close()

#### feed
sess.run([output], feed_dict={input1:[7.], input2:[2.]})

#### device
with tf.device("/gpu:1")
