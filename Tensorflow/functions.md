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
*TensorFlow provides several operations and classes that you can use to control the execution of operations and add conditional dependencies to your graph*
- tf.identity(input, name=None) <br/>
`Describe: Return a tensor with the same shape and contents as the input tensor or value`
- tf.tuple(tensors, name=None, control_inputs=None) <br/>
`Describe: Group tensors together`
- tf.group(\*inputs, \*\*kwargs) <br/>
`Describe: Create an op that groups multiple operations`
- tf.no_op(name=None) <br/>
`Describe: Does nothing. Only useful as a placeholder for control edges`
- tf.count_up_to(ref, limit, name=None) <br/>
`Describe: Increments 'ref' until it reaches 'limit'`
##### Logical Operators
*TensorFlow provides several operations that you can use to add logical operators to your graph*
- tf.logical_and(x, y, name=None) <br/>
`Describe: Returns the truth value of x AND y element-wise`
- tf.logical_not(x, name=None) <br/>
`Describe: Returns the truth value of NOT x element-wise`
- tf.logical_or(x, y, name=None) <br/>
`Describe: Returns the truth value of x OR y element-wise`
- tf.logical_xor(x, y, name='LogicalXor')
##### Comparison Operators
*TensorFlow provides several operations that you can use to add comparison operators to your graph*
- tf.equal(x, y, name=None) <br/>
`Describe: Returns the truth value of (x == y) element-wise`
- tf.not_equal(x, y, name=None) <br/>
`Describe: Returns the truth value of (x != y) element-wise`
- tf.less(x, y, name=None) <br/>
`Describe: Returns the truth value of (x < y) element-wise`
- tf.less_equal(x, y, name=None) <br/>
`Describe: Returns the truth value of (x <= y) element-wise`
- tf.greater(x, y, name=None) <br/>
`Describe: Returns the truth value of (x > y) element-wise`
- tf.greater_equal(x, y, name=None) <br/>
`Describe: Returns the truth value of (x >= y) element-wise`
- tf.select(condition, t, e, name=None) <br/>
`Describe: Selects elements from t or e, depending on condition`
- tf.where(condition, x=None, y=None, name=None) <br/>
`Describe: Return the elements, either from x or y, depending on the condition`
##### Debugging Operations
*TensorFlow provides several operations that you can use to validate values and debug your graph*
- tf.is_finite(x, name=None) <br/>
`Describe: Returns which elements of x are finite`
- tf.is_inf(x, name=None) <br/>
`Describe: Returns which elements of x are Inf`
- tf.is_nan(x, name=None) <br/>
`Describe: Returns which elements of x are NaN`
- tf.verify_tensor_all_finite(t, msg, name=None) <br/>
`Describe: Assert that the tensor does not contain any NaN's or Inf's`
- tf.check_numerics(tensor, message, name=None) <br/>
`Describe: Checks a tensor for NaN and Inf values`
- tf.add_check_numerics_ops() <br/>
`Describe: Connect a check_numerics to every floating point tensor`
- tf.Assert(condition, data, summarize=None, name=None) <br/>
`Describe: Asserts that the given condition is true`
- tf.Print(input_, data, message=None, first_n=None, summarize=None, name=None) <br/>
`Describe: Prints a list of tensors`

### Framwork
Building Graphs
*Classes and functions for building TensorFlow graphs*
##### Core graph data structures
- class tf.Graph <br/>
`Describe: A TensorFlow computation, represented as a dataflow graph`
  - tf.Graph.\_\_init\_\_()
  - tf.Graph.as_default()
  - tf.Graph.as_graph_def(from_version=None)
  - tf.Graph.finalize()
  - tf.Graph.finalized
  - tf.Graph.control_dependencies(control_inputs)
  - tf.Graph.device(device_name_or_function)
  - tf.Graph.name_scope(name)
  - tf.Graph.add_to_collection(name, value)
  - tf.Graph.get_collection(name, scope=None)
  - tf.Graph.as_graph_element(obj, allow_tensor=True, allow_operation=True)
  - tf.Graph.get_operation_by_name(name)
  - tf.Graph.get_tensor_by_name(name)
  - tf.Graph.get_operations()
  - tf.Graph.get_default_device()
  - tf.Graph.seed
  - tf.Graph.unique_name(name)
  - tf.Graph.version
  - tf.Graph.create_op(op_type, inputs, dtypes, input_types=None, name=None, attrs=None, op_def=None, compute_shapes=True)
  - tf.Graph.gradient_override_map(op_type_map)  
- class tf.Operation <br/>
`Describe: Represents a graph node that performs computation on tensors`
  - tf.Operation.name
  - tf.Operation.type
  - tf.Operation.inputs
  - tf.Operation.control_inputs
  - tf.Operation.outputs
  - tf.Operation.device
  - tf.Operation.graph
  - tf.Operation.run(feed_dict=None, session=None)
  - tf.Operation.get_attr(name)
  - tf.Operation.traceback
  - tf.Operation.__init__(node_def, g, inputs=None, output_types=None, control_inputs=None, input_types=None, original_op=None, op_def=None)
  - tf.Operation.node_def
  - tf.Operation.op_def
  - tf.Operation.values()
- class tf.Tensor <br/>
`Describe: Represents a value produced by an Operation`
  - tf.Tensor.dtype
  - tf.Tensor.name
  - tf.Tensor.value_index
  - tf.Tensor.graph
  - tf.Tensor.op
  - tf.Tensor.consumers()
  - tf.Tensor.eval(feed_dict=None, session=None) <br/>
  `Describe: Evaluates this tensor in a Session`
  - tf.Tensor.get_shape()
  - tf.Tensor.set_shape(shape)
  - tf.Tensor.__init__(op, value_index, dtype)
  - tf.Tensor.device
##### Tensor types
- class tf.DType <br/>
`Describe: Represents the type of the elements in a Tensor`
  - tf.DType.is_compatible_with(other)
  - tf.DType.name
  - tf.DType.base_dtype
  - tf.DType.is_ref_dtype
  - tf.DType.as_ref
  - tf.DType.is_integer
  - tf.DType.is_quantized
  - tf.DType.as_numpy_dtype
  - tf.DType.as_datatype_enum
  - tf.DType.__init__(type_enum)
  - tf.DType.max
  - tf.DType.min
  - tf.as_dtype(type_value) 
- tf.as_dtype(type_value) <br/>
`Describe: Converts the given type_value to a DType`
##### Utility functions
- tf.device(dev)
- tf.name_scope(name)
- tf.control_dependencies(control_inputs)
- tf.convert_to_tensor(value, dtype=None, name=None) <br/>
`Describe: Converts the given value to a Tensor`
- tf.get_default_graph() <br/>
`Describe: Returns the default graph for the current thread`
- tf.import_graph_def(graph_def, input_map=None, return_elements=None, name=None, op_dict=None) <br/>
`Describe: Imports the TensorFlow graph in graph_def into the Python Graph`
##### Graph collections
- tf.add_to_collection(name, value)
- tf.get_collection(key, scope=None)
- class tf.GraphKeys <br/>
`Describe: Standard names to use for graph collections`
  - VARIABLES
  - TRAINABLE_VARIABLES
  - SUMMARIES
  - QUEUE_RUNNERS
##### Defining new operations
- class tf.RegisterGradient <br/>
`Describe: A decorator for registering the gradient function for an op type`
  - tf.RegisterGradient.\_\_init\_\_(op_type)
- tf.NoGradient(op_type) <br/>
`Describe: Specifies that ops of type op_type do not have a defined gradient`
- class tf.RegisterShape <br/>
`Describe: A decorator for registering the shape function for an op type`
  - tf.RegisterShape.\_\_init\_\_(op_type)
- class tf.TensorShape <br/>
`Describe: Represents the shape of a Tensor`
  - tf.TensorShape.merge_with(other)
  - tf.TensorShape.concatenate(other)
  - tf.TensorShape.ndims
  - tf.TensorShape.dims
  - tf.TensorShape.as_list()
  - tf.TensorShape.is_compatible_with(other)
  - tf.TensorShape.is_fully_defined()
  - tf.TensorShape.with_rank(rank)
  - tf.TensorShape.with_rank_at_least(rank)
  - tf.TensorShape.with_rank_at_most(rank)
  - tf.TensorShape.assert_has_rank(rank)
  - tf.TensorShape.assert_same_rank(other)
  - tf.TensorShape.assert_is_compatible_with(other)
  - tf.TensorShape.assert_is_fully_defined()
  - tf.TensorShape.\_\_init\_\_(dims)
  - tf.TensorShape.as_dimension_list()
  - tf.TensorShape.num_elements()
- class tf.Dimension <br/>
`Describe: Represents the value of one dimension in a TensorShape`
  - tf.Dimension.\_\_init\_\_(value)
  - tf.Dimension.assert_is_compatible_with(other)
  - tf.Dimension.is_compatible_with(other)
  - tf.Dimension.merge_with(other)
  - tf.Dimension.value
- tf.op_scope(values, name, default_name) <br/>
`Describe: Returns a context manager for use when defining a Python op`
- tf.get_seed(op_seed) <br/>
`Describe: Returns the local seeds an operation should use given an op-specific seed`

### Image
#### Encoding and Decoding
*TensorFlow provides Ops to decode and encode JPEG and PNG formats. Encoded images are represented by scalar string Tensors, decoded images by 3-D uint8 tensors of shape \[height, width, channels\]*
- tf.image.decode_jpeg(contents, channels=None, ratio=None, fancy_upscaling=None, try_recover_truncated=None, acceptable_fraction=None, name=None) <br/>
`Describe: Decode a JPEG-encoded image to a uint8 tensor`
- tf.image.encode_jpeg(image, format=None, quality=None, progressive=None, optimize_size=None, chroma_downsampling=None, density_unit=None, x_density=None, y_density=None, xmp_metadata=None, name=None) <br/>
`Describe: JPEG-encode an image`
- tf.image.decode_png(contents, channels=None, name=None) <br/>
`Describe: Decode a PNG-encoded image to a uint8 tensor`
- tf.image.encode_png(image, compression=None, name=None) <br/>
`Describe: PNG-encode an image`
#### Resizing
*The resizing Ops accept input images as tensors of several types. They always output resized images as float32 tensors*
- tf.image.resize_images(images, new_height, new_width, method=0) <br/>
`Describe: Resize images to new_width, new_height using the specified method`
- tf.image.resize_area(images, size, name=None) <br/>
`Describe: Resize images to size using area interpolation`
- tf.image.resize_bicubic(images, size, name=None) <br/>
`Describe: Resize images to size using bicubic interpolation`
- tf.image.resize_bilinear(images, size, name=None) <br/>
`Describe: Resize images to size using bilinear interpolation`
- tf.image.resize_nearest_neighbor(images, size, name=None) <br/>
`Describe: Resize images to size using nearest neighbor interpolation`
#### Cropping
- tf.image.resize_image_with_crop_or_pad(image, target_height, target_width) <br/>
`Describe: Crops and/or pads an image to a target width and height`
- tf.image.pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width) <br/>
`Describe: Pad image with zeros to the specified height and width`
- tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width) <br/>
`Describe: Crops an image to a specified bounding box`
- tf.image.random_crop(image, size, seed=None, name=None) <br/>
`Describe: Randomly crops image to size [target_height, target_width]`
- tf.image.extract_glimpse(input, size, offsets, centered=None, normalized=None, uniform_noise=None, name=None) <br/>
`Describe: Extracts a glimpse from the input tensor`
#### Flipping and Transposing
- tf.image.flip_up_down(image) <br/>
`Describe: Flip an image horizontally (upside down)`
- tf.image.random_flip_up_down(image, seed=None) <br/>
`Describe: Randomly flips an image vertically (upside down)`
- tf.image.flip_left_right(image) <br/>
`Describe: Flip an image horizontally (left to right)`
- tf.image.random_flip_left_right(image, seed=None) <br/>
`Describe: Randomly flip an image horizontally (left to right)`
- tf.image.transpose_image(image) <br/>
`Describe: Transpose an image by swapping the first and second dimension`
#### Image Adjustments
*TensorFlow provides functions to adjust images in various ways: brightness, contrast, hue, and saturation. Each adjustment can be done with predefined parameters or with random parameters picked from predefined intervals. Random adjustments are often useful to expand a training set and reduce overfitting*
- tf.image.adjust_brightness(image, delta, min_value=None, max_value=None) <br/>
`Describe: Adjust the brightness of RGB or Grayscale images`
- tf.image.random_brightness(image, max_delta, seed=None) <br/>
`Describe: Adjust the brightness of images by a random factor`
- tf.image.adjust_contrast(images, contrast_factor, min_value=None, max_value=None) <br/>
`Describe: Adjust contrast of RGB or grayscale images`
- tf.image.random_contrast(image, lower, upper, seed=None) <br/>
`Describe: Adjust the contrase of an image by a random factor`
- tf.image.per_image_whitening(image) <br/>
`Describe: Linearly scales image to have zero mean and unit norm`

### IO
Inputs and Readers
#### Placeholders
*TensorFlow provides a placeholder operation that must be fed with data on execution*
- tf.placeholder(dtype, shape=None, name=None) <br/>
`Describe: Inserts a placeholder for a tensor that will be always fed`
#### Readers
*TensorFlow provides a set of Reader classes for reading data formats*
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
- tf.subtract(x, y, name=None)
- tf.multiply(x, y, name=None) <br/>
`Describe: Returns x * y element-wise`
- tf.div(x, y, name=None)
- tf.mod(x, y, name=None)
#### Basic Math Functions
- tf.add_n(inputs, name=None) <br/>
`Describe: Add all input tensors element wise`
- tf.abs(x, name=None)
- tf.negative(x, name=None)
- tf.sign(x, name=None) <br/>
`Describe: Returns an element-wise indication of the sign of a number` <br/>
```python
y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0
```
- tf.inv(x, name=None)
- tf.square(x, name=None)
- tf.round(x, name=None)
- tf.sqrt(x, name=None)
- tf.rsqrt(x, name=None)
- tf.pow(x, y, name=None) <br/>
`Describe: Computes the power of one value to another`
- tf.exp(x, name=None)
- tf.log(x, name=None)
- tf.ceil(x, name=None) <br/>
`Describe: Returns element-wise smallest integer in not less than x`
- tf.floor(x, name=None) <br/>
`Describe: Returns element-wise largest integer not greater than x`
- tf.maximum(x, y, name=None) <br/>
`Describe: Returns the max of x and y (i.e. x > y ? x : y) element-wise, broadcasts`
- tf.minimum(x, y, name=None) <br/>
`Describe: Returns the min of x and y (i.e. x < y ? x : y) element-wise, broadcasts`
- tf.cos(x, name=None)
- tf.sin(x, name=None)
#### Matrix Math Functions
*TensorFlow provides several operations that you can use to add basic mathematical functions for matrices to your graph*
- tf.diag(diagonal, name=None) <br/>
`Describe: Returns a diagonal tensor with a given diagonal values`
- tf.transpose(a, perm=None, name='transpose') <br/>
`Describe: Transposes a. Permutes the dimensions according to perm`
- tf.matmul(a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None) <br/>
`Describe: Multiplies matrix a by matrix b, producing a * b`
- tf.batch_matmul(x, y, adj_x=None, adj_y=None, name=None) <br/>
`Describe: Multiplies slices of two tensors in batches`
- tf.matrix_determinant(input, name=None) <br/>
`Describe: Calculates the determinant of a square matrix`
- tf.batch_matrix_determinant(input, name=None) <br/>
`Describe: Calculates the determinants for a batch of square matrices`
- tf.matrix_inverse(input, name=None)
- tf.batch_matrix_inverse(input, name=None)
- tf.cholesky(input, name=None) <br/>
`Describe: Calculates the Cholesky decomposition of a square matrix`
- tf.batch_cholesky(input, name=None)  <br/>
`Describe: Calculates the Cholesky decomposition of a batch of square matrices`
#### Complex Number Functions
- tf.complex(real, imag, name=None)
- tf.complex_abs(x, name=None)
- tf.conj(in_, name=None)
- tf.imag(in_, name=None)
- tf.real(in_, name=None)
#### Reduction
*TensorFlow provides several operations that you can use to perform common math computations that reduce various dimensions of a tensor*
- tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None) <br/>
`Describe: Computes the sum of elements across dimensions of a tensor`
- tf.reduce_prod(input_tensor, reduction_indices=None, keep_dims=False, name=None) <br/>
`Describe: Computes the product of elements across dimensions of a tensor`
- tf.reduce_min(input_tensor, reduction_indices=None, keep_dims=False, name=None) <br/>
`Describe: Computes the minimum of elements across dimensions of a tensor`
- tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None) <br/>
`Describe: Computes the maximum of elements across dimensions of a tensor`
- tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None) <br/>
`Describe: Computes the mean of elements across dimensions of a tensor`
- tf.reduce_all(input_tensor, reduction_indices=None, keep_dims=False, name=None) <br/>
`Describe: Computes the "logical and" of elements across dimensions of a tensor`
- tf.reduce_any(input_tensor, reduction_indices=None, keep_dims=False, name=None) <br/>
`Describe: Computes the "logical or" of elements across dimensions of a tensor`
- tf.accumulate_n(inputs, shape=None, tensor_dtype=None, name=None) <br/>
`Describe: Returns the element-wise sum of a list of tensors`
#### Segmentation
*TensorFlow provides several operations that you can use to perform common math computations on tensor segments*
- tf.segment_sum(data, segment_ids, name=None)
- tf.segment_prod(data, segment_ids, name=None)
- tf.segment_min(data, segment_ids, name=None)
- tf.segment_max(data, segment_ids, name=None)
- tf.segment_mean(data, segment_ids, name=None)
- tf.unsorted_segment_sum(data, segment_ids, num_segments, name=None)
- tf.sparse_segment_sum(data, indices, segment_ids, name=None)
- tf.sparse_segment_mean(data, indices, segment_ids, name=None)
#### Sequence Comparison and Indexing
*TensorFlow provides several operations that you can use to add sequence comparison and index extraction to your graph. You can use these operations to determine sequence differences and determine the indexes of specific values in a tensor*
- tf.argmin(input, dimension, name=None) <br/>
`Describe: Returns the index with the smallest value across dimensions of a tensor`
- tf.argmax(input, dimension, name=None) <br/>
`Describe: Returns the index with the largest value across dimensions of a tensor`
- tf.listdiff(x, y, name=None) <br/>
`Describe: Computes the difference between two lists of numbers`
- tf.where(input, name=None) <br/>
`Describe: Returns locations of true values in a boolean tensor`
- tf.unique(x, name=None) <br/>
`Describe: Finds unique elements in a 1-D tensor`
- tf.edit_distance(hypothesis, truth, normalize=True, name='edit_distance') <br/>
`Describe: Computes the Levenshtein distance between sequences`
- tf.invert_permutation(x, name=None) <br/>
`Describe: Computes the inverse permutation of a tensor`

### Neural Network
#### Activation Functions
*The activation ops provide different types of nonlinearities for use in neural networks*
- tf.nn.relu(features, name=None) <br/>
`Describe: Computes rectified linear: max(features, 0)`
- tf.nn.relu6(features, name=None) <br/>
`Describe: Computes Rectified Linear 6: min(max(features, 0), 6)`
- tf.nn.softplus(features, name=None) <br/>
`Describe: Computes softplus: log(exp(features) + 1)`
- tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None) <br/>
`Describe: Computes dropout`
- tf.nn.bias_add(value, bias, name=None) <br/>
`Describe: Adds bias to value (This is (mostly) a special case of tf.add where bias is restricted to 1-D)`
- tf.sigmoid(x, name=None) <br/>
`Describe: Computes sigmoid of x element-wise (Specifically, y = 1 / (1 + exp(-x)))`
- tf.tanh(x, name=None) <br/>
`Describe: Computes hyperbolic tangent of x element-wise`
#### Convolution
*The convolution ops sweep a 2-D filter over a batch of images, applying the filter to each window of each image of the appropriate size*
- tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None) <br/>
`Describe: Computes a 2-D convolution given 4-D input and filter tensors` <br/>
`input shape: [batch, in_height, in_width, in_channels], filter shape: [filter_height, filter_width, in_channels, out_channels]`
- tf.nn.depthwise_conv2d(input, filter, strides, padding, name=None) <br/>
`Describe: Depthwise 2-D convolution`
- tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, name=None) <br/>
`Describe: 2-D convolution with separable filters`
#### Pooling
*The pooling ops sweep a rectangular window over the input tensor, computing a reduction operation for each window (average, max, or max with argmax)*
- tf.nn.avg_pool(value, ksize, strides, padding, name=None) <br/>
`Describe: Performs the average pooling on the input`
- tf.nn.max_pool(value, ksize, strides, padding, name=None) <br/>
`Describe: Performs the max pooling on the input`
- tf.nn.max_pool_with_argmax(input, ksize, strides, padding, Targmax=None, name=None) <br/>
`Describe: Performs max pooling on the input and outputs both max values and indices`
#### Normalization
*Normalization is useful to prevent neurons from saturating when inputs may have varying scale, and to aid generalization*
- tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None) <br/>
`Describe: Normalizes along dimension dim using an L2 norm`
- tf.nn.local_response_normalization(input, depth_radius=None, bias=None, alpha=None, beta=None, name=None) <br/>
`Describe: Local Response Normalization`
- tf.nn.moments(x, axes, name=None) <br/>
`Describe: Calculate the mean and variance of x`
#### Losses
*The loss ops measure error between two tensors, or between a tensor and zero. These can be used for measuring accuracy of a network in a regression task or for regularization purposes (weight decay)*
- tf.nn.l2_loss(t, name=None) <br/>
`Describe: L2 Loss (sum(t ** 2) / 2)`
- tf.losses.sigmoid_cross_entropy
- tf.losses.softmax_cross_entropy
- tf.losses.huber_loss
#### Classification
*TensorFlow provides several operations that help you perform classification*
- tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None) <br/>
`Describe: Computes sigmoid cross entropy given logits`
- tf.nn.softmax(logits, name=None) <br/>
`Describe: Computes softmax activations`
- tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None) <br/>
`Describe: Computes softmax cross entropy between logits and labels`
#### Embeddings
*TensorFlow provides library support for looking up values in embedding tensors*
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
  - tf.Variable.\_\_init__(initial_value, trainable=True, collections=None, validate_shape=True, name=None) <br/>
  `Describe: Creates a new variable with value initial_value`
  - tf.Variable.initialized_value() <br/>
  `Describe: Returns the value of the initialized variable`
  - tf.Variable.assign(value, use_locking=False) <br/>
  `Describe: Assigns a new value to the variable`
  - tf.Variable.assign_add(delta, use_locking=False) <br/>
  `Describe: Adds a value to this variable`
  - tf.Variable.assign_sub(delta, use_locking=False) <br/>
  `Describe: Subtracts a value from this variable`
  - tf.Variable.scatter_sub(sparse_delta, use_locking=False) <br/>
  `Describe: Subtracts IndexedSlices from this variable`
  - tf.Variable.count_up_to(limit) <br/>
  `Describe: Increments this variable until it reaches limit`
  - tf.Variable.eval(session=None) <br/>
  `Describe: In a session, computes and returns the value of this variable`
  - tf.Variable.name <br/>
  `Describe: The name of this variable`
  - tf.Variable.dtype <br/>
  `Describe: The DType of this variable`
  - tf.Variable.get_shape() <br/>
  `Describe: The TensorShape of this variable`
  - tf.Variable.device <br/>
  `Describe: The device of this variable`
  - tf.Variable.initializer <br/>
  `Describe: The initializer operation for this variable`
  - tf.Variable.graph <br/>
  `Describe: The Graph of this variable`
  - tf.Variable.op <br/>
  `Describe: The Operation of this variable`
#### Variable helper functions
*TensorFlow provides a set of functions to help manage the set of variables collected in the graph*
- tf.all_variables() <br>
`Describe: Returns all variables collected in the graph`
- tf.trainable_variables() <br>
`Describe: Returns all variables created with trainable=True`
- tf.initialize_all_variables() <br>
`Describe: Returns an Op that initializes all variables`
- tf.initialize_variables(var_list, name='init') <br>
`Describe: Returns an Op that initializes a list of variables`
- tf.assert_variables_initialized(var_list=None) <br>
`Describe: Returns an Op to check if variables are initialized`
#### Saving and Restoring Variables
- class tf.train.Saver <br/>
`Describe: Saves and restores variables`
  - tf.train.Saver.__init__(var_list=None, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False, saver_def=None, builder=None) <br/>
  `Describe: Creates a Saver`
  - tf.train.Saver.save(sess, save_path, global_step=None, latest_filename=None) <br/>
  `Describe: Saves variables`
  - tf.train.Saver.restore(sess, save_path) <br/>
  `Describe: Restores previously saved variables`
  - tf.train.Saver.last_checkpoints <br/>
  `Describe: List of not-yet-deleted checkpoint filenames`
  - tf.train.Saver.set_last_checkpoints(last_checkpoints) <br/>
  `Describe: Sets the list of not-yet-deleted checkpoint filenames`
  - tf.train.Saver.as_saver_def() <br/>
  `Describe: Generates a SaverDef representation of this saver`
- tf.train.latest_checkpoint(checkpoint_dir, latest_filename=None) <br/>
`Describe: Finds the filename of latest saved checkpoint file`
- tf.train.get_checkpoint_state(checkpoint_dir, latest_filename=None) <br/>
`Describe: Returns CheckpointState proto from the "checkpoint" file`
- tf.train.update_checkpoint_state(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None, latest_filename=None) <br/>
`Describe: Updates the content of the 'checkpoint' file`
#### Sharing Variables
*TensorFlow provides several classes and operations that you can use to create variables contingent on certain conditions*
- tf.get_variable(name, shape=None, dtype=tf.float32, initializer=None, trainable=True, collections=None) <br/>
`Describe: Gets an existing variable with these parameters or create a new one`
- tf.get_variable_scope() <br/>
`Describe: Returns the current variable scope`
- tf.variable_scope(name_or_scope, reuse=None, initializer=None) <br/>
`Describe: Returns a context for variable scope`
- tf.constant_initializer(value=0.0) <br/>
`Describe: Returns an initializer that generates Tensors with a single value`
- tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None) <br/>
`Describe: Returns an initializer that generates Tensors with a normal distribution`
- tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None) <br/>
`Describe: Returns an initializer that generates a truncated normal distribution`
- tf.random_uniform_initializer(minval=0.0, maxval=1.0, seed=None) <br/>
`Describe: Returns an initializer that generates Tensors with a uniform distribution`
- tf.uniform_unit_scaling_initializer(factor=1.0, seed=None) <br/>
`Describe: Returns an initializer that generates tensors without scaling variance`
- tf.zeros_initializer(shape, dtype=tf.float32) <br/>
`Describe: An adaptor for zeros() to match the Initializer spec`
#### Sparse Variable Updates
- tf.scatter_update(ref, indices, updates, use_locking=None, name=None)
- tf.scatter_add(ref, indices, updates, use_locking=None, name=None)
- tf.scatter_sub(ref, indices, updates, use_locking=None, name=None)
- tf.sparse_mask(a, mask_indices, name=None)
- class tf.IndexedSlices
  - tf.IndexedSlices.__init__(values, indices, dense_shape=None)
  - tf.IndexedSlices.values
  - tf.IndexedSlices.indices
  - tf.IndexedSlices.dense_shape
  - tf.IndexedSlices.name
  - tf.IndexedSlices.dtype
  - tf.IndexedSlices.device
  - tf.IndexedSlices.op

### Train
*This library provides a set of classes and functions that helps train models*
#### Optimizers
*The Optimizer base class provides methods to compute gradients for a loss and apply gradients to variables. A collection of subclasses implement classic optimization algorithms such as GradientDescent and Adagrad*
- class tf.train.Optimizer <br/>
`Describe: Base class for optimizers`
  - tf.train.Optimizer.\_\_init__(use_locking, name) <br/>
  `Describe: Create a new Optimizer`
  - tf.train.Optimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, name=None) <br/>
  `Describe: Add operations to minimize 'loss' by updating 'var_list'`
  - tf.train.Optimizer.compute_gradients(loss, var_list=None, gate_gradients=1) <br/>
  `Describe: Compute gradients of "loss" for the variables in "var_list"`
  - tf.train.Optimizer.apply_gradients(grads_and_vars, global_step=None, name=None) <br/>
  `Describe: Apply gradients to variables`
  - tf.train.Optimizer.get_slot_names() <br/>
  `Describe: Return a list of the names of slots created by the Optimizer`
  - tf.train.Optimizer.get_slot(var, name) <br/>
  `Describe: Return a slot named "name" created for "var" by the Optimizer`
- class tf.train.GradientDescentOptimizer <br/>
`Describe: Optimizer that implements the gradient descent algorithm`
  - tf.train.GradientDescentOptimizer.\_\_init__(learning_rate, use_locking=False, name='GradientDescent') <br/>
  `Describe: Construct a new gradient descent optimizer`
- class tf.train.AdagradOptimizer <br/>
`Describe: Optimizer that implements the Adagrad algorithm`
  - tf.train.AdagradOptimizer.\_\_init__(learning_rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad') <br/>
  `Describe: Construct a new Adagrad optimizer`
- class tf.train.MomentumOptimizer <br/>
`Describe: Optimizer that implements the Momentum algorithm`
  - tf.train.MomentumOptimizer.\_\_init__(learning_rate, momentum, use_locking=False, name='Momentum') <br/>
  `Describe: Construct a new Momentum optimizer`
- class tf.train.AdamOptimizer <br/>
`Describe: Optimizer that implements the Adam algorithm`
  - tf.train.AdamOptimizer.\_\_init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam') <br/>
  `Describe: Construct a new Adam optimizer`
- class tf.train.FtrlOptimizer <br/>
`Describe: Optimizer that implements the FTRL algorithm`
  - tf.train.FtrlOptimizer.__init__(learning_rate, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='Ftrl') <br/>
  `Describe: Construct a new FTRL optimizer`
- class tf.train.RMSPropOptimizer <br/>
`Describe: Optimizer that implements the RMSProp algorithm`
  - tf.train.RMSPropOptimizer.__init__(learning_rate, decay, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp') <br/>
  `Describe: Construct a new RMSProp optimizer`
#### Gradient Computation
*TensorFlow provides functions to compute the derivatives for a given TensorFlow computation graph, adding operations to the graph. The optimizer classes automatically compute derivatives on your graph, but creators of new Optimizers or expert users can call the lower-level functions below*
- tf.gradients(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None) <br/>
`Describe: Constructs symbolic partial derivatives of ys w.r.t. x in xs`
- class tf.AggregationMethod <br/>
`Describe: A class listing aggregation methods used to combine gradients`
- tf.stop_gradient(input, name=None) <br/>
`Describe: Stops gradient computation`
#### Gradient Clipping
*TensorFlow provides several operations that you can use to add clipping functions to your graph*
- tf.clip_by_value(t, clip_value_min, clip_value_max, name=None) <br/>
`Describe: Clips tensor values to a specified min and max (Any values less than clip_value_min are set to clip_value_min. Any values greater than clip_value_max are set to clip_value_max)`
- tf.clip_by_norm(t, clip_norm, name=None) <br/>
`Describe: Clips tensor values to a maximum L2-norm`
- tf.clip_by_average_norm(t, clip_norm, name=None) <br/>
`Describe: Clips tensor values to a maximum average L2-norm`
- tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None) <br/>
`Describe: Clips values of multiple tensors by the ratio of the sum of their norms`
- tf.global_norm(t_list, name=None) <br/>
`Describe: Computes the global norm of multiple tensors`
#### Decaying the learning rate
- tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None) <br/>
`Describe: Applies exponential decay to the learning rate`
#### Moving Averages
*Some training algorithms, such as GradientDescent and Momentum often benefit from maintaining a moving average of variables during optimization. Using the moving averages for evaluations often improve results significantly*
- class tf.train.ExponentialMovingAverage <br/>
`Describe: Maintains moving averages of variables by employing and exponential decay`
  - tf.train.ExponentialMovingAverage.\_\_init__(decay, num_updates=None, name='ExponentialMovingAverage')
  - tf.train.ExponentialMovingAverage.apply(var_list=None)
  - tf.train.ExponentialMovingAverage.average_name(var)
  - tf.train.ExponentialMovingAverage.average(var)
#### Coordinator and QueueRunner
- class tf.train.Coordinator
  - tf.train.Coordinator.__init__()
  - tf.train.Coordinator.join(threads, stop_grace_period_secs=120)
  - tf.train.Coordinator.request_stop(ex=None)
  - tf.train.Coordinator.should_stop()
  - tf.train.Coordinator.wait_for_stop(timeout=None)
- class tf.train.QueueRunner
  - tf.train.QueueRunner.__init__(queue, enqueue_ops)
  - tf.train.QueueRunner.create_threads(sess, coord=None, daemon=False, start=False)
  - tf.train.QueueRunner.exceptions_raised
- tf.train.add_queue_runner(qr, collection='queue_runners')
- tf.train.start_queue_runners(sess=None, coord=None, daemon=True, start=True, collection='queue_runners')
#### Summary Operations
*The following ops output Summary protocol buffers as serialized string tensors*
- tf.summary.scalar(name, tensor, collections=None, family=None) <br/>
`Describe: Outputs a Summary protocol buffer with scalar values` <br/>
`Example: tf.summary.scalar("loss", loss)`
- tf.summary.image(name, tensor, max_outputs=3, collections=None, family=None) <br/>
`Describe: Outputs a Summary protocol buffer with images (tensor must be 4-D with shape [batch_size, height, width, channels])`
- tf.summary.histogram(name, values, collections=None, family=None) <br/>
`Describe: Outputs a Summary protocol buffer with a histogram`
- tf.nn.zero_fraction(value, name=None) <br/>
`Describe: Returns the fraction of zeros in value`
- tf.summary.merge(inputs, collections=None, name=None) <br/>
`Describe: Merges summaries`
- tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES, scope=None, name=None) <br/>
`Describe: Merges all summaries collected in the default graph`
#### Adding Summaries to Event Files
*Writes Summary protocol buffers to event files*
- class tf.summary.FileWritter <br/>
  - tf.summary.FileWriter.\_\_init__(self, logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None)
  `Describe: Creates a SummaryWriter and an event file`
  - tf.summary.FileWriter.add_summary(summary, global_step=None) <br/>
  `Describe: Adds a Summary protocol buffer to the event file`
  - tf.summary.FileWriter.add_event(event) <br/>
  `Describe: Adds an event to the event file`
  - tf.summary.FileWriter.add_graph(graph_def, global_step=None) <br/>
  `Describe: Adds a GraphDef protocol buffer to the event file`
  - tf.summary.FileWriter.flush() <br/>
  `Describe: Flushes the event file to disk`
  - tf.summary.FileWriter.close() <br/>
  `Describe: Flushes the event file to disk and close the file`
- tf.train.summary_iterator(path) <br/>
`Describe: An iterator for reading Event protocol buffers from an event file`
#### Training utilities
- tf.train.global_step(sess, global_step_tensor) <br/>
`Describe: Small helper to get the global step`
- tf.train.write_graph(graph_def, logdir, name, as_text=True) <br/>
`Describe: Writes a graph proto on disk`

### Other
- tf.newaxis










