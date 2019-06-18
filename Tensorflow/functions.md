# Tensorflow Functions

- Table of Contents
  - Maths
  - Array
  - Matrix
  - Neuronal Network
  - Checkpointing
  - Queues and syncronizations
  - Flow control
  
### Data Definition

- tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
- tf.placeholder(dtype, shape=None, name=None)
- tf.Variable
 
### Maths

#### Arithmetic Operation
 
operation: Add, Subtract, Multiply, Div, Exp, Log, Greater, Less, Equal
 
- tf.add(x, y, name=None)
- tf.subtract(x, y, name=None)
- tf.multiply(x, y, name=None)
- tf.scalar_mul(scalar, x)
- tf.div(x, y, name=None)
- tf.mod(x, y, name=None)
- tf.abs(x, name=None)
- tf.negative(x, name=None)
- tf.sign(x, name=None) <br />
`Example: y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0`
- tf.inv(x, name=None)
- tf.square(x, name=None)
- tf.round(x, name=None)
- tf.sqrt(x, name=None)
- tf.pow(x, y, name=None)
- tf.exp(x, name=None)
- tf.log(x, name=None)
- tf.maximum(x, y, name=None)
- tf.minimum(x, y, name=None)
- tf.cos(x, name=None)
- tf.sin(x, name=None)
- tf.tan(x, name=None)
- tf.atan(x, name=None)

### Array

#### Tensor Transformations

- tf.string_to_number(string_tensor, out_type=None, name=None)
- tf.to_double(x, name=’ToDouble’)
- tf.to_float(x, name=’ToFloat’)
- tf.to_int32(x, name=’ToInt32’)
- tf.to_int64(x, name=’ToInt64’)
- tf.cast(x, dtype, name=None)

#### Shapes and Shaping

- tf.shape(input, name=None)
- tf.size(input, name=None)
- tf.rank(input, name=None)
- tf.reshape(tensor, shape, name=None)
- tf.expand_dims(input, dim, name=None)

#### Slicing and Joining

- tf.slice(input_, begin, size, name=None)
- tf.split(split_dim, num_split, value, name=’split’)
- tf.concat(concat_dim, values, name=’concat’)
- tf.pack(values, axis=0, name=’pack’)
- tf.reverse(tensor, dims, name=None)
- tf.transpose(a, perm=None, name=’transpose’)
- tf.gather(params, indices, validate_indices=None, name=None)
- tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)

### Matrix

#### Matrix Operation

- tf.diag(diagonal, name=None)
- tf.diag_part(input, name=None)
- tf.trace(x, name=None)
- tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
- tf.matrix_determinant(input, name=None)
- tf.matrix_inverse(input, adjoint=None, name=None)
- tf.cholesky(input, name=None)
- tf.matrix_solve(matrix, rhs, adjoint=None, name=None)

#### Complex

- tf.complex(real, imag, name=None)
- tf.complex_abs(x, name=None)
- tf.conj(input, name=None)
- tf.imag(input, name=None)
- tf.real(input, name=None)
- tf.fft(input, name=None)

#### Reduction

- tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.reduce_prod(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.reduce_min(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.reduce_all(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.reduce_any(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.accumulate_n(inputs, shape=None, tensor_dtype=None, name=None)
- tf.cumsum(x, axis=0, exclusive=False, reverse=False, name=None)

#### Segmentation

- tf.segment_sum(data, segment_ids, name=None)
- tf.segment_prod(data, segment_ids, name=None)
- tf.segment_min(data, segment_ids, name=None)
- tf.segment_max(data, segment_ids, name=None)
- tf.segment_mean(data, segment_ids, name=None)
- tf.unsorted_segment_sum(data, segment_ids, num_segments, name=None)
- tf.sparse_segment_sum(data, indices, segment_ids, name=None)

#### Sequence Comparison and Indexing

- tf.argmin(input, dimension, name=None)
- tf.argmax(input, dimension, name=None)
- tf.listdiff(x, y, name=None)
- tf.where(input, name=None)
- tf.unique(x, name=None)	
- tf.invert_permutation(x, name=None)

### Neural Network

#### Activation Functions

- tf.nn.relu(features, name=None)
- tf.nn.relu6(features, name=None)
- tf.nn.elu(features, name=None)
- tf.nn.softplus(features, name=None)
- tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
- tf.nn.bias_add(value, bias, data_format=None, name=None)
- tf.sigmoid(x, name=None)	y = 1 / (1 + exp(-x))
- tf.tanh(x, name=None)

#### Convolution

- tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None) <br />
`input shape: [batch_size, in_height, in_width, in_channels]`
- tf.nn.conv3d(input, filter, strides, padding, name=None) <br />
`input shape: [batch_size, in_depth, in_height, in_width, in_channels]`

#### Pooling

- tf.nn.avg_pool(value, ksize, strides, padding, data_format=’NHWC’, name=None)
- tf.nn.max_pool(value, ksize, strides, padding, data_format=’NHWC’, name=None)
- tf.nn.max_pool_with_argmax(input, ksize, strides, padding, Targmax=None, name=None)
- tf.nn.avg_pool3d(input, ksize, strides, padding, name=None)
- tf.nn.max_pool3d(input, ksize, strides, padding, name=None)

#### Normalization

- tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)
- tf.nn.sufficient_statistics(x, axes, shift=None, keep_dims=False, name=None)
- tf.nn.normalize_moments(counts, mean_ss, variance_ss, shift, name=None)
- tf.nn.moments(x, axes, shift=None, name=None, keep_dims=False)

#### Losses

- tf.nn.l2_loss(t, name=None)
- tf.losses.sigmoid_cross_entropy
- tf.losses.softmax_cross_entropy
- tf.losses.huber_loss

#### Classification

- tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)
- tf.nn.softmax(logits, name=None)
- tf.nn.log_softmax(logits, name=None)
- tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
- tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)
- tf.nn.weighted_cross_entropy_with_logits(logits, targets, pos_weight, name=None)

#### Embeddings

- tf.nn.embedding_lookup(params, ids, partition_strategy=’mod’, name=None, validate_indices=True)
- tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights, partition_strategy=’mod’, name=None, combiner=’mean’)

#### Recurrent Neural Networks

- tf.nn.rnn(cell, inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)
- tf.nn.dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None)
- tf.nn.state_saving_rnn(cell, inputs, state_saver, state_name, sequence_length=None, scope=None)
- tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs, initial_state_fw=None, initial_state_bw=None, dtype=None, sequence_length=None, scope=None)

#### Evaluation

- tf.nn.top_k(input, k=1, sorted=True, name=None)
- tf.nn.in_top_k(predictions, targets, k, name=None)

#### Candidate Sampling

- tf.nn.nce_loss(weights, biases, inputs, labels, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=False, partition_strategy=’mod’, name=’nce_loss’)
- tf.nn.sampled_softmax_loss(weights, biases, inputs, labels, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=True, partition_strategy=’mod’, name=’sampled_softmax_loss’)
- tf.nn.uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)
- tf.nn.log_uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)
- tf.nn.learned_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)
- tf.nn.fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, vocab_file=”, distortion=1.0, num_reserved_ids=0, num_shards=1, shard=0, unigrams=(), seed=None, name=None)

### Saving and Restoring Variables
 
tf.train.Saver.__init__(var_list=None, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False, saver_def=None, builder=None)
	
- tf.train.Saver.save(sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix=’meta’, write_meta_graph=True)
- tf.train.Saver.restore(sess, save_path)
- tf.train.Saver.last_checkpoints
- tf.train.Saver.set_last_checkpoints(last_checkpoints)
- tf.train.Saver.set_last_checkpoints_with_time(last_checkpoints_with_time)


 

- tf.tile

- tf.train.AdamOptimizer

- tf.clip_by_value

- tf.pad

- tf.convert_to_tensor

- tf.stop_gradient

- tf.newaxis

- tf.contrib.slim.conv2d

- tf.contrib.slim.max_pool2d

- tf.contrib.slim.fully_connected

- tf.contrib.rnn.GRUCell







