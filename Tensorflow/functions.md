# Tensorflow Functions

- Table of Contents
  - Array
  - Client
  - Constant
  - Control Flow
  - Framework
  - Image
  - IO
  - Math
  - Neuronal Network
  - State
  - Train
  
### Array
*Note: Functions taking Tensor arguments can also take anything accepted by tf.convert_to_tensor* <br/>
Tensor Transformations
##### Casting
*TensorFlow provides several operations that you can use to cast tensor data types in your graph*
- tf.string_to_number(string_tensor, out_type=None, name=None)
- tf.to_double(x, name='ToDouble')
- tf.to_float(x, name='ToFloat')
- tf.to_bfloat16(x, name='ToBFloat16')
- tf.to_int32(x, name='ToInt32')
- tf.to_int64(x, name='ToInt64')
- tf.cast(x, dtype, name=None) <br/>
`Describe: Casts a tensor to a new type`
##### Shapes and Shaping
*TensorFlow provides several operations that you can use to determine the shape of a tensor and change the shape of a tensor*
- tf.shape(input, name=None) <br/>
`Describe: returns a 1-D integer tensor representing the shape of input`
- tf.size(input, name=None) <br/>
`Describe: returns an integer representing the number of elements in input`
- tf.rank(input, name=None) <br/>
`Describe: returns an integer representing the rank of input` <br/>
`Rank: The rank of a tensor is not the same as the rank of a matrix. The rank of a tensor is the number of indices required to uniquely select each element of the tensor. Rank is also known as "order", "degree", or "ndims."`
- tf.reshape(tensor, shape, name=None) <br/>
`Describe: returns a tensor that has the same values as tensor with shape shape` <br/>
`If shape is the special value [-1], then tensor is flattened and the operation outputs a 1-D tensor with all elements of tensor.`
`If shape is 1-D or higher, then the operation returns a tensor with shape shape filled with the values of tensor. In this case, the number of elements implied by shape must be the same as the number of elements in tensor.` <br/>
- tf.squeeze(input, squeeze_dims=None, name=None) <br/>
`Describe: Removes dimensions of size 1 from the shape of a tensor`
- tf.expand_dims(input, dim, name=None) <br/>
`Describe: Inserts a dimension of 1 into a tensor's shape`
##### Slicing and Joining
*TensorFlow provides several operations to slice or extract parts of a tensor, or join multiple tensors together*
- tf.slice(input_, begin, size, name=None) <br/>
`Describe: Extracts a slice from a tensor`
- tf.split(split_dim, num_split, value, name='split') <br/>
`Describe: Splits a tensor into num_split tensors along one dimension`
- tf.tile(input, multiples, name=None) <br/>
`Describe: Constructs a tensor by tiling a given tensor`
- tf.pad(input, paddings, name=None) <br/>
`Describe: Pads a tensor with zeros`
- tf.concat(concat_dim, values, name='concat') <br/>
`Describe: Concatenates tensors along one dimension`
- tf.stack(values, axis=0, name='stack') <br/>
`Describe: Packs a list of rank-R tensors into one rank-(R+1) tensor`
- tf.unstack(value, num=None, axis=0, name='unstack') <br/>
`Describe: Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors`
- tf.reverse_sequence(input, seq_lengths, seq_axis=None, batch_axis=None, name=None, seq_dim=None, batch_dim=None) <br/>
`Describe: Reverses variable length slices in dimension seq_dim`
- tf.reverse(tensor, dims, name=None) <br/>
`Describe: Reverses specific dimensions of a tensor` <br/>
`Note: Given a tensor, and a bool tensor dims representing the dimensions of tensor, this operation reverses each dimension i of tensor where dims[i] is True`
- tf.transpose(a, perm=None, name='transpose')
- tf.gather(params, indices, validate_indices=None, name=None, axis=0) <br/>
`Describe: Gather slices from params according to indices`
- tf.dynamic_partition(data, partitions, num_partitions, name=None) <br/>
`Describe: Partitions data into num_partitions tensors using indices from partitions`
- tf.dynamic_stitch(indices, data, name=None) <br/>
`Describe: Interleave the values from the data tensors into a single tensor`

### Client
Running Graphs
*This library contains classes for launching graphs and executing operations*
#### Session management
- class tf.Session <br/>
`Describe: A class for running TensorFlow operations`
  - tf.Session.run(fetches, feed_dict=None)
  - tf.Session.close()
  - tf.Session.graph
  - tf.Session.as_default()
``` python
sess = tf.Session()   |    with tf.Session() as sess:     |    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
sess.run(...)         |      sess.run(...)                |                                            log_device_placement=True)) 
sess.close()          |                                   |
```
- class tf.InteractiveSession <br/>
`Describe: A TensorFlow Session for use in interactive contexts, such as a shell`
  - tf.InteractiveSession.close()
``` python
sess = tf.InteractiveSession()
Tensor.eval()
Operation.run()
sess.close()
```
- tf.get_default_session()
#### Error classes
- class tf.OpError
- class tf.errors.CancelledError
- class tf.errors.UnknownError
- class tf.errors.InvalidArgumentError
- class tf.errors.DeadlineExceededError
- class tf.errors.NotFoundError
- class tf.errors.AlreadyExistsError
- class tf.errors.PermissionDeniedError
- class tf.errors.UnauthenticatedError
- class tf.errors.ResourceExhaustedError
- class tf.errors.FailedPreconditionError
- class tf.errors.AbortedError
- class tf.errors.OutOfRangeError
- class tf.errors.UnimplementedError
- class tf.errors.InternalError
- class tf.errors.UnavailableError
- class tf.errors.DataLossError

### Constant
Constants, Sequences, and Random Values
##### Constant Value Tensors
*TensorFlow provides several operations that you can use to generate constants*
- tf.zeros(shape, dtype=tf.float32, name=None)
- tf.zeros_like(tensor, dtype=None, name=None)
- tf.ones(shape, dtype=tf.float32, name=None)
- tf.ones_like(tensor, dtype=None, name=None)
- tf.fill(dims, value, name=None) <br/>
`Describe: Creates a tensor filled with a scalar value`
- tf.constant(value, dtype=None, shape=None, name='Const')
##### Sequences
- tf.linspace(start, stop, num, name=None) <br/>
`Describe: Generates values in an interval`
- tf.range(start, limit, delta=1, name='range') <br/>
`Describe: Creates a sequence of integers`
##### Random Tensors
*TensorFlow has several ops that create random tensors with different distributions. The random ops are stateful, and create new random values each time they are evaluated*
- tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) <br/>
`Desceibe: Outputs random values from a normal distribution`
- tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) <br/>
`Desceibe: Outputs random values from a truncated normal distribution`
- tf.random_uniform(shape, minval=0.0, maxval=1.0, dtype=tf.float32, seed=None, name=None) <br/>
`Describe: Outputs random values from a uniform distribution`
- tf.random_shuffle(value, seed=None, name=None) <br/>
`Describe: Randomly shuffles a tensor along its first dimension`
- tf.set_random_seed(seed) <br/>
`Describe: Sets the graph-level random seed`

### Control Flow
##### Control Flow Operations
- tf.identity(input, name=None)
- tf.tuple(tensors, name=None, control_inputs=None)
- tf.group(*inputs, **kwargs)
- tf.no_op(name=None)
- tf.count_up_to(ref, limit, name=None)
##### Logical Operators
- tf.logical_and(x, y, name=None)
- tf.logical_not(x, name=None)
- tf.logical_or(x, y, name=None)
- tf.logical_xor(x, y, name='LogicalXor')
##### Comparison Operators
- tf.equal(x, y, name=None)
- tf.not_equal(x, y, name=None)
- tf.less(x, y, name=None)
- tf.less_equal(x, y, name=None)
- tf.greater(x, y, name=None)
- tf.greater_equal(x, y, name=None)
- tf.select(condition, t, e, name=None)
- tf.where(input, name=None)
##### Debugging Operations
- tf.is_finite(x, name=None)
- tf.is_inf(x, name=None)
- tf.is_nan(x, name=None)
- tf.verify_tensor_all_finite(t, msg, name=None)
- tf.check_numerics(tensor, message, name=None)
- tf.add_check_numerics_ops()
- tf.Assert(condition, data, summarize=None, name=None)
- tf.Print(input_, data, message=None, first_n=None, summarize=None, name=None)

### Framwork
Building Graphs
##### Core graph data structures
- class tf.Graph
- class tf.Operation
- class tf.Tensor
##### Tensor types
- class tf.DType
- tf.as_dtype(type_value)
##### Utility functions
- tf.device(dev)
- tf.name_scope(name)
- tf.control_dependencies(control_inputs)
- tf.convert_to_tensor(value, dtype=None, name=None)
- tf.get_default_graph()
- tf.import_graph_def(graph_def, input_map=None, return_elements=None, name=None, op_dict=None)
##### Graph collections
- tf.add_to_collection(name, value)
- tf.get_collection(key, scope=None)
- class tf.GraphKeys
##### Defining new operations
- class tf.RegisterGradient
- tf.NoGradient(op_type)
- class tf.RegisterShape
- class tf.TensorShape
- class tf.Dimension
- tf.op_scope(values, name, default_name)
- tf.get_seed(op_seed)

### Image
#### Encoding and Decoding
- tf.image.decode_jpeg(contents, channels=None, ratio=None, fancy_upscaling=None, try_recover_truncated=None, acceptable_fraction=None, name=None)
- tf.image.encode_jpeg(image, format=None, quality=None, progressive=None, optimize_size=None, chroma_downsampling=None, density_unit=None, x_density=None, y_density=None, xmp_metadata=None, name=None)
- tf.image.decode_png(contents, channels=None, name=None)
- tf.image.encode_png(image, compression=None, name=None)
#### Resizing
- tf.image.resize_images(images, new_height, new_width, method=0)
- tf.image.resize_area(images, size, name=None)
- tf.image.resize_bicubic(images, size, name=None)
- tf.image.resize_bilinear(images, size, name=None)
- tf.image.resize_nearest_neighbor(images, size, name=None)
#### Cropping
- tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)
- tf.image.pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
- tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
- tf.image.random_crop(image, size, seed=None, name=None)
- tf.image.extract_glimpse(input, size, offsets, centered=None, normalized=None, uniform_noise=None, name=None)
#### Flipping and Transposing
- tf.image.flip_up_down(image)
- tf.image.random_flip_up_down(image, seed=None)
- tf.image.flip_left_right(image)
- tf.image.random_flip_left_right(image, seed=None)
- tf.image.transpose_image(image)
#### Image Adjustments
- tf.image.adjust_brightness(image, delta, min_value=None, max_value=None)
- tf.image.random_brightness(image, max_delta, seed=None)
- tf.image.adjust_contrast(images, contrast_factor, min_value=None, max_value=None)
- tf.image.random_contrast(image, lower, upper, seed=None)
- tf.image.per_image_whitening(image)

### IO
Inputs and Readers
#### Placeholders
- tf.placeholder(dtype, shape=None, name=None)
#### Readers
- class tf.ReaderBase
- class tf.TextLineReader
- class tf.WholeFileReader
- class tf.IdentityReader
- class tf.TFRecordReader
- class tf.FixedLengthRecordReader
#### Converting
- tf.decode_csv(records, record_defaults, field_delim=None, name=None)
- tf.decode_raw(bytes, out_type, little_endian=None, name=None)
##### Example protocol buffer
- tf.parse_example(serialized, names=None, sparse_keys=None, sparse_types=None, dense_keys=None, dense_types=None, dense_defaults=None, dense_shapes=None, name='ParseExample')
- tf.parse_single_example(serialized, names=None, sparse_keys=None, sparse_types=None, dense_keys=None, dense_types=None, dense_defaults=None, dense_shapes=None, name='ParseSingleExample')
#### Queues
- class tf.QueueBase
- class tf.FIFOQueue
- class tf.RandomShuffleQueue
#### Dealing with the filesystem
- tf.matching_files(pattern, name=None)
- tf.read_file(filename, name=None)
#### Input pipeline
##### Beginning of an input pipeline
- tf.train.match_filenames_once(pattern, name=None)
- tf.train.limit_epochs(tensor, num_epochs=None, name=None)
- tf.train.range_input_producer(limit, num_epochs=None, shuffle=True, seed=None, capacity=32, name=None)
- tf.train.slice_input_producer(tensor_list, num_epochs=None, shuffle=True, seed=None, capacity=32, name=None)
- tf.train.string_input_producer(string_tensor, num_epochs=None, shuffle=True, seed=None, capacity=32, name=None)
##### Batching at the end of an input pipeline
- tf.train.batch(tensor_list, batch_size, num_threads=1, capacity=32, enqueue_many=False, shapes=None, name=None)
- tf.train.batch_join(tensor_list_list, batch_size, capacity=32, enqueue_many=False, shapes=None, name=None)
- tf.train.shuffle_batch(tensor_list, batch_size, capacity, min_after_dequeue, num_threads=1, seed=None, enqueue_many=False, shapes=None, name=None)
- tf.train.shuffle_batch_join(tensor_list_list, batch_size, capacity, min_after_dequeue, seed=None, enqueue_many=False, shapes=None, name=None)

### Math
#### Arithmetic Operators
- tf.add(x, y, name=None)
- tf.sub(x, y, name=None)
- tf.mul(x, y, name=None)
- tf.div(x, y, name=None)
- tf.mod(x, y, name=None)
#### Basic Math Functions
- tf.add_n(inputs, name=None)
- tf.abs(x, name=None)
- tf.neg(x, name=None)
- tf.sign(x, name=None)
- tf.inv(x, name=None)
- tf.square(x, name=None)
- tf.round(x, name=None)
- tf.sqrt(x, name=None)
- tf.rsqrt(x, name=None)
- tf.pow(x, y, name=None)
- tf.exp(x, name=None)
- tf.log(x, name=None)
- tf.ceil(x, name=None)
- tf.floor(x, name=None)
- tf.maximum(x, y, name=None)
- tf.minimum(x, y, name=None)
- tf.cos(x, name=None)
- tf.sin(x, name=None)
#### Matrix Math Functions
- tf.diag(diagonal, name=None)
- tf.transpose(a, perm=None, name='transpose')
- tf.matmul(a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
- tf.batch_matmul(x, y, adj_x=None, adj_y=None, name=None)
- tf.matrix_determinant(input, name=None)
- tf.batch_matrix_determinant(input, name=None)
- tf.matrix_inverse(input, name=None)
- tf.batch_matrix_inverse(input, name=None)
- tf.cholesky(input, name=None)
- tf.batch_cholesky(input, name=None)
#### Complex Number Functions
- tf.complex(real, imag, name=None)
- tf.complex_abs(x, name=None)
- tf.conj(in_, name=None)
- tf.imag(in_, name=None)
- tf.real(in_, name=None)
#### Reduction
- tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.reduce_prod(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.reduce_min(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.reduce_all(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.reduce_any(input_tensor, reduction_indices=None, keep_dims=False, name=None)
- tf.accumulate_n(inputs, shape=None, tensor_dtype=None, name=None)
#### Segmentation
- tf.segment_sum(data, segment_ids, name=None)
- tf.segment_prod(data, segment_ids, name=None)
- tf.segment_min(data, segment_ids, name=None)
- tf.segment_max(data, segment_ids, name=None)
- tf.segment_mean(data, segment_ids, name=None)
- tf.unsorted_segment_sum(data, segment_ids, num_segments, name=None)
- tf.sparse_segment_sum(data, indices, segment_ids, name=None)
- tf.sparse_segment_mean(data, indices, segment_ids, name=None)
#### Sequence Comparison and Indexing
- tf.argmin(input, dimension, name=None)
- tf.argmax(input, dimension, name=None)
- tf.listdiff(x, y, name=None)
- tf.where(input, name=None)
- tf.unique(x, name=None)
- tf.edit_distance(hypothesis, truth, normalize=True, name='edit_distance')
- tf.invert_permutation(x, name=None)

### Neural Network
#### Activation Functions
- tf.nn.relu(features, name=None)
- tf.nn.relu6(features, name=None)
- tf.nn.softplus(features, name=None)
- tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
- tf.nn.bias_add(value, bias, name=None)
- tf.sigmoid(x, name=None)
- tf.tanh(x, name=None)
#### Convolution
- tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
- tf.nn.depthwise_conv2d(input, filter, strides, padding, name=None)
- tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, name=None)
#### Pooling
- tf.nn.avg_pool(value, ksize, strides, padding, name=None)
- tf.nn.max_pool(value, ksize, strides, padding, name=None)
- tf.nn.max_pool_with_argmax(input, ksize, strides, padding, Targmax=None, name=None)
#### Normalization
- tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)
- tf.nn.local_response_normalization(input, depth_radius=None, bias=None, alpha=None, beta=None, name=None)
- tf.nn.moments(x, axes, name=None)
#### Losses
- tf.nn.l2_loss(t, name=None)
- tf.losses.sigmoid_cross_entropy
- tf.losses.softmax_cross_entropy
- tf.losses.huber_loss
#### Classification
- tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)
- tf.nn.softmax(logits, name=None)
- tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
#### Embeddings
- tf.nn.embedding_lookup(params, ids, name=None)
#### Evaluation
- tf.nn.top_k(input, k, name=None)
- tf.nn.in_top_k(predictions, targets, k, name=None)
#### Candidate Sampling
##### Sampled Loss Functions
- tf.nn.nce_loss(weights, biases, inputs, labels, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=False, name='nce_loss')
- tf.nn.sampled_softmax_loss(weights, biases, inputs, labels, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=True, name='sampled_softmax_loss')
##### Candidate Samplers
- tf.nn.uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)
- tf.nn.log_uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)
- tf.nn.learned_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)
- tf.nn.fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, vocab_file='', distortion=0.0, num_reserved_ids=0, num_shards=1, shard=0, unigrams=\[\], seed=None, name=None)
##### Miscellaneous candidate sampling utilities
- tf.nn.compute_accidental_hits(true_classes, sampled_candidates, num_true, seed=None, name=None)

### State
#### Variables
- class tf.Variable
#### Variable helper functions
- tf.all_variables()
- tf.trainable_variables()
- tf.initialize_all_variables()
- tf.initialize_variables(var_list, name='init')
- tf.assert_variables_initialized(var_list=None)
#### Saving and Restoring Variables
- class tf.train.Saver
- tf.train.latest_checkpoint(checkpoint_dir, latest_filename=None)
- tf.train.get_checkpoint_state(checkpoint_dir, latest_filename=None)
- tf.train.update_checkpoint_state(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None, latest_filename=None)
#### Sharing Variables
- tf.get_variable(name, shape=None, dtype=tf.float32, initializer=None, trainable=True, collections=None)
- tf.get_variable_scope()
- tf.variable_scope(name_or_scope, reuse=None, initializer=None)
- tf.constant_initializer(value=0.0)
- tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None)
- tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None)
- tf.random_uniform_initializer(minval=0.0, maxval=1.0, seed=None)
- tf.uniform_unit_scaling_initializer(factor=1.0, seed=None)
- tf.zeros_initializer(shape, dtype=tf.float32)
#### Sparse Variable Updates
- tf.scatter_update(ref, indices, updates, use_locking=None, name=None)
- tf.scatter_add(ref, indices, updates, use_locking=None, name=None)
- tf.scatter_sub(ref, indices, updates, use_locking=None, name=None)
- tf.sparse_mask(a, mask_indices, name=None)
- class tf.IndexedSlices

### Train
#### Optimizers
- class tf.train.GradientDescentOptimizer
- class tf.train.AdagradOptimizer
- class tf.train.MomentumOptimizer
- class tf.train.AdamOptimizer
- class tf.train.FtrlOptimizer
- class tf.train.RMSPropOptimizer
#### Gradient Computation
- tf.gradients(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)
class tf.AggregationMethod
- tf.stop_gradient(input, name=None)
#### Gradient Clipping
- tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)
- tf.clip_by_norm(t, clip_norm, name=None)
- tf.clip_by_average_norm(t, clip_norm, name=None)
- tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)
- tf.global_norm(t_list, name=None)
#### Decaying the learning rate
- tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
#### Moving Averages
- class tf.train.ExponentialMovingAverage
#### Coordinator and QueueRunner
- class tf.train.Coordinator
- class tf.train.QueueRunner
- tf.train.add_queue_runner(qr, collection='queue_runners')
- tf.train.start_queue_runners(sess=None, coord=None, daemon=True, start=True, collection='queue_runners')
#### Summary Operations
- tf.scalar_summary(tags, values, collections=None, name=None)
- tf.image_summary(tag, tensor, max_images=None, collections=None, name=None)
- tf.histogram_summary(tag, values, collections=None, name=None)
- tf.nn.zero_fraction(value, name=None)
- tf.merge_summary(inputs, collections=None, name=None)
- tf.merge_all_summaries(key='summaries')
#### Adding Summaries to Event Files
- class tf.train.SummaryWriter
- tf.train.summary_iterator(path)
#### Training utilities
- tf.train.global_step(sess, global_step_tensor)
- tf.train.write_graph(graph_def, logdir, name, as_text=True)





- tf.newaxis

- tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)









