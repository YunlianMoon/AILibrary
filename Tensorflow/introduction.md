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

(1)
sess = tf.Session()
sess.run(init)
fetch:
sess.run()
(2)
sess = tf.InteractiveSession()
sess.run(init)
fetch:
Tensor.eval()
Operation.run()
(3)
with tf.Session() as sess:
  sess.run(init)

sess.close()

#### feed
sess.run([output], feed_dict={input1:[7.], input2:[2.]})

#### device
with tf.device("/gpu:1")
