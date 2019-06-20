# TensorFlow

import tensorflow as tf

### Basic Usage

#### source op
- tf.placeholder(dtype, shape=None, name=None)

- tf.Variable(initializer, name)
- tf.zeros(shape, dtype=tf.float32, name=None)
- tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
- tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
- tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
- tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)


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
  
config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

sess.close()

#### feed
sess.run([output], feed_dict={input1:[7.], input2:[2.]})

#### device
with tf.device("/gpu:1")

#### summary

- tf.scalar_summary(loss.op.name, loss)

tf.merge_all_summaries()

summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

summary_str = sess.run(summary_op, feed_dict=feed_dict)

summary_writer.add_summary(summary_str, step)


