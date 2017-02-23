# Index
- [Graph and Session notions](#graph-and-session)
- [Dynamic and Static shape](#dynamic-shape-vs-static-shape)
- [Tensorflow variable](#variable)
- [Structure tensorflow code with decorator](#how-to-structure-a-tensorflow-model)
- [Save and restore model](#checkpoint)
- [Tensoarboard](#tensoarboard)
- [Regularization](#regularization)
- [Preprocessing](#preprocessing)
- [CNN & shits](#computer-vision-application)
- [RNN & shits](#nlp-application)
- [Higher order operations](#higher-order-operators)
- [Debugging](#debugging)
- [Miscellanous](#miscellanous)
- [Dynamic graph computation](#tensorflow-fold)
- [Tensorflow Estimator](#tensorflow-estimator)

# Graph and Session
### Graph vs Session
* A graph defines the computation. It doesn’t compute anything, it doesn’t hold any values, it just defines the operations that you specified in your code.
* A session allows to execute graphs or part of graphs. It allocates resources (on one or more machines) for that and holds the actual values of intermediate results and variables. One can create a session with tf.Session, and be sure to use a context manager, or tf.Session.close(), because all ressources of the session are saved. To run some graph element, you should use the function .run(graph_element, feed_dic), it return values, or list of values, if a list of graph elements was passed.

### Interactive session
Interactive session is usefull when multiple different sessions needed to be run in a same script
```
start the session
sess = tf.InteractiveSession()

# stop the session
sess.stop()
ops.reset_default_graph()
```

### Collect variable in the graph
To collect and retrieve vales associated with a graph, it is possible to get them with GraphKeys. For example ```GLOBAL_VARIABLE```, or ```MODEL_VARIABLE```, or ```TRAINABLE_VARIABLE```, ```QUEUE_RUNNERS```, or even more specifically the ```WEIGHTS```, ```BIASES```, or ```ACTIVATIONS```.

* You can get the name of all variables that have not been initialized by passing a list of Variable to the function ```tf.report_uninitialized_variables(list_var). It returns a list of names of uninitialized variables

### Split training variables between multiple neural networks
Imagine two neural networks, created in two different scopes, one in "generator" scope, and one in "discriminator" scope.  
1. First you need to retrieve all trainable variable with ```train_variables = tf.train_variables()```  
2. Then you split the training variable in two lists:  
```
list_gen = self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
list_dis = self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]	
```  
3. Create two functions for training them:  
```
grads = optimizer.compute_gradients(loss_generator, var_list=list_gen)
train_gen = optimizer.apply_gradients(grads)
```

### Get an approximation of the size in byte of the graph
```
for v in tf.global_variables():
	vars += np.prod(v.get_shape().as_list())
vars *= 4
```

# Dynamic shape vs Static shape
For getting the dynamic shape, running a session is necessary, because ds = ss where all None are replaced by the session dimension. Example
```
var = tf.placeholder(shape=(None, 2), dtype=tf.float32)
dynamic_shape = tf.shape(var)
print(var.get_shape().as_list())  # => [None, 2]
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(dynamic_shape, feed_dict={var: [[1, 2], [1, 2]]}))  # => [2 2]
```

# Variable
## Some parameters
* Setting trainable=False keeps the variable out of the GraphKeys.TRAINABLE_VARIABLES collection in the graph, so they won't be trained when backpropagating. 
* Setting collections=[] keeps the variable out of the GraphKeys.GLOBAL_VARIABLES collection used for saving and restoring checkpoints.  
Example: ```input_data = tf.Variable(data_initializer, trainable=False, collections=[])```

## Shared variable
It is possible to reuse weights, just by setting new variable to older one defined previously. For that, you must be in the same namescope, and look for a variable with the same name. Here is an example:
```
def build():
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    ...

def build_funct():
     with tf.variable_scope("scope1"):
	relu1 = build()
     with tf.variable_scope("scope2"):
	# relu2 is different from relu1, even if they shared the same name, they are in different namescope	
	relu2 = build()

# however calling twice the build_funct() will return an error


result1 = build_funct()
result2 = build_funct()
# Raises ValueError(... scope1/weights already exists ...) because
# f.get_variable_scope().reuse == False 

# to avoid the error, you must defined this way:
with tf.variable_scope("image_filters") as scope:
	result1 = build_funct()
	scope.reuse_variables()
	result2 = build_funct()
```

# How to structure a Tensorflow model
```python
import functools
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class Model:
    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.prediction
        self.optimize
        self.error
	# define all placeholders here

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        x = self.image
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 10, tf.nn.softmax)
        return x

    @define_scope
    def optimize(self):
        logprob = tf.log(self.prediction + 1e-12)
        cross_entropy = -tf.reduce_sum(self.label * logprob)
        optimizer = tf.train.RMSPropOptimizer(0.03)
        return optimizer.minimize(cross_entropy)

    @define_scope
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    model = Model(image, label)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for _ in range(10):
      images, labels = mnist.test.images, mnist.test.labels
      error = sess.run(model.error, {image: images, label: labels})
      print('Test error {:6.2f}%'.format(100 * error))
      for _ in range(60):
        images, labels = mnist.train.next_batch(100)
        sess.run(model.optimize, {image: images, label: labels})


if __name__ == '__main__':
    main()
```

# Checkpoint
### Save and Restore
```
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=5) # max number of checkpoint to save
last_saved_model = tf.train.latest_checkpoint('./')
# Restore
if last_saved_model is not None:
    saver.restore(sess, last_saved_model)

# Save during the training session
if itr % 100 == 0:
    saver.save(sess, global_step=itr, save_path="model")
```

### What are the files saved
* The checkpoint file is use in combination of high level helper	for different time loading saved chkg
* The meta chkp hold the compressed Protobuf graph of your model and all te metadata associated
* The chkp file holds the data
* The events file store everything for visualization

### Connect graph already trained
It is possible to connect multiple graph, for example if you want to connect vgg19 to a new graph, and only trained the last one, here is a simple example:
```
vgg_saver = tf.train.import_meta_graph(dir + '/vgg/results/vgg-16.meta')
vgg_graph = tf.get_default_graph()

# retrieve inputs
self.x_plh = vgg_graph.get_tensor_by_name('input:0')

# choose the node to connect to 
output_conv =vgg_graph.get_tensor_by_name('conv1_2:0')

# stop the gradient for fine tuning
output_conv_sg = tf.stop_gradient(output_conv)

# create your own neural network
output_conv_shape = output_conv_sg.get_shape().as_list()
"""..."""
```

# Tensoarboard
* Launch a session with ```tensorboard --logdir=""```.

### Save the graph for Tensorflow visualization
The FileWriter class provides a mechanism to create an event file in a given directory and add summaries and events to it. The class updates is called asynchronously, which means it will never slow down the training loop calling.
```python
sess = tf.Session()
summary_writer = tf.summary.FileWriter('logs', graph=sess.graph)
```
Connection between them is done with the line: ```with tf.Session(graph=graph) as sess:```
### Summary about activation and gradient
```
def add_activation_summary(var):
    tf.summary.histogram(var.op.name + "/activation", var)
    tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)
```

### Summary about an image
Automatic rescaling of float value to value between [0, 255]  
```
tf.summary.image([batch_size, height, width, channels], max_output= num_max_of_images_to_display)```
```

### Summary about a cost function
```
cost_function = -tf.reduce_sum(var)
# don't need to store the reference, next function is responsible to collect all summaries
tf.summary.scalar("cost_function", cross_entropy)
```

### Summary about a python scalar
It is also possible to add a summary for a python scalar
```
summary_accuracy = tf.Summary(value=[
            tf.Summary.Value(tag="value_name", simple_value=value_to_save),
        ])
```
You can call this function as many time as you want, if you call it with the same tag, no duplicate value will be saved. You can even plot a graph of this python value, by writing the ```summary_accuracy``` every epoch.

### Merge all summaries operations
```python
merged_summary_op = tf.summary.merge_all()
```
If you create new summary after this function, they won't be collect by calling ```merged_summary_op```.

### Collect stat during each iteration
```python
summary_str, _ = sess.run([merged_summary_op, optimize], {x: batchX, y: batchY})
summary_writer.add_summary(summary_str, current_iter (must be unique id))
```

### Plot embeddings

1. Create an embedding vector (dim: nb_embeddings, embedding_size)
    ```python
    embedding = tf.Variable(tf.random_normal([nb_embedding, embedding_size]))
    ```

2. Create a tag for every embedding (first name in the file correspond to name of the first embedding  
    ```	
    LOG_DIR = 'log/'
    metadata = os.path.join(LOG_DIR, 'metadata.tsv')
    with open(metadata, 'w') as metadata_file:
        for name in whatever_object:
            metadata_file.write('%s\n' % name)
    ```

3. Save embedding
    ```    
    # See more advance tuto on Saver object
    saver = tf.train.Saver([movie_embedding])
    saver.save(sess, os.path.join(LOG_DIR, 'movie_embeddings.ckpt'))
    ```

4. Create a projector for Tensorboard
    ```
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding.name # embedding is the tf.Variable()
    embedding.metadata_path = metadata # metadata is a filename
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
    ```
    

# Regularization
### L2 regularization
```python
w = tf.Variable()
cost = # define your loss
regularizer = tf.nn.l2_loss(w)
loss = cost + regularizer
```
### L1 and L2 regularization
```
def l1_l2_regularizer(weight_l1=1.0, weight_l2=1.0, scope=None):
    """
    L1 and L2 regularizer
    :param weight_l1:
    :param weight_l2:
    :param scope:
    :return:
    """
    def regularizer(tensor):
        with tf.name_scope(scope, 'L1L2Regularizer', [tensor]):
            weight_l1_t = tf.convert_to_tensor(weight_l1,
                                               dtype=tensor.dtype.base_dtype,
                                               name='weight_l1')
            weight_l2_t = tf.convert_to_tensor(weight_l2,
                                               dtype=tensor.dtype.base_dtype,
                                               name='weight_l2')
            reg_l1 = tf.multiply(weight_l1_t, tf.reduce_sum(tf.abs(tensor)),
                                 name='value_l1')
            reg_l2 = tf.multiply(weight_l2_t, tf.nn.l2_loss(tensor),
                                 name='value_l2')
            return tf.add(reg_l1, reg_l2, name='value')

    return regularizer
```
### Dropout
```
hidden_layer_drop = tf.nn.dropout(some_activation_output, keep_prob)
```

### Batch normalization
Use the tf.nn.contrib.layers.batch_norm
```
is_training = tf.placeholder(tf.bool)
batch_norm(pre_activation, is_training=is_training, scale=True)
```  
By default, movingmean, and movingscale are not in the default graph, but in _updateOperation, hence to compute the movingmean, and movingscale, you should that the operation should be computed before the loss function is calculated:

```
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# puis par exemple
with tf.control_dependencies(update_ops):
            grads_dis = self.optimizer.compute_gradients(loss=self.dis_loss, var_list=self.dis_variables)
            self.train_dis = self.optimizer.apply_gradients(grads_dis)
```

The trainable boolean can be a placeholder, so that depending on the feeding dictionnary, the computation in the batch norm layer will be different

# Preprocessing
It is possible to load data directly from numpy arrays, however it is best practise to use ```tf.SequenceExample```. It is very verbose, but allows reusability, and really split between the model and the data preprocessing  

1. Create a function to transform a batch element to a ```SequenceExample```:  
    ```python
    def make_example(inputs, labels):
    	ex = tf.train.SequenceExample()
    	# Add non-sequential feature
    	seq_len = len(inputs)
    	ex.context.feature["length"].int64_list.value.append(sequence_length)

    	# Add sequential feature
    	fl_labels = ex.feature_lists.feature_list["labels"]
    	fl_tokens = ex.feature_lists.feature_list["inputs"]
    	for token, label in zip(inputs, labels):
    		fl_labels.feature.add().int64_list.value.append(label)
    		fl_tokens.feature.add().int64_list.value.append(token)
    	return ex
    ```

2. Write all example into TFRecords  
    ```
    import tempfile
    with tempfile.NamedTemporaryFile() as fp:
    	writer = tf.python_io.TFRecordWriter(fp.name)
    	for input, label_sequence in zip(all_inputs, all_labels):
    		ex = make_example(input, label_sequence)
    		writer.write(ex.SerializeToString())
    	writer.close()
    	# check where file is writen with fp.name
    ```
3. Retrieve file with TFRecordReader in an object named ```ex```.
4. Define how to parse the data
    ```python
    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }

    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=ex,
        context_features=context_features,
        sequence_features=sequence_features
    )
    ```
5. Retrieve the data into array  
    ```python

    # get back in array format
    context = tf.contrib.learn.run_n(context_parsed, n=1, feed_dict=None)
    ```

# Computer vision application
## Convolution
Reminder on simple convolution:
* Given an input of channel size |k| equal 1, the neuron (one output) of a feature map (a channel, let's say channel 1 in the output) is the result given by a filter W1 apply to some location in the single input channel. Then you stride the same filter over the input image, and compute another input for the same output channel. Every output channel has its own filter.  
* Now if the input channel size |k| is superieur than 1, for each output channel, there as k filter (kernel). Each of theses filter is applied respectively on every input channel location and then sum (not mean) to give the value of a neuron. Hence the number of parameters in a convolution is ```|filter_height * filter_width * nb_input_channels * nb_output_channels|```. This is also why its difficult for the first layer of a convolution neural network to catch high level features because usually the input channel size is small, and hence, the information for an output neuron, wasn't computed with a lot of filter. In deeper layer, usually output neuron in a given channel are computed by summing over a lot of filters, hence each filter can capture different representations.  
Implementation:
```
# 5*5 conv, 1 input_channel_size, output_channel_size
W = tf.Variable(tf.random_normal([5, 5, 1, 32]))
# dimension of x is [batch_size, 28, 28, 1]
x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
```

## Transpose convolution
There is nothing fancy in the formula of the transpose convolution, the only trick is that the input channel is transformed (padded) with respect to the size of the filter and the dimension of the output. Here is a nice example of deconvolutions: [deconvolution without no stride](https://i.stack.imgur.com/YyCu2.gif), and [deconvolution with stride](https://i.stack.imgur.com/f2RiP.gif).  
Implementation:
```
# h2 is of shape [self.batch_size, 7, 7, 128]
output_shape_h3 = [self.batch_size, 14, 14, 64]
# filter_size, output_channel_size, input_channel_size
W3 = utils.weight_variable([5, 5, 64, 128], name="W3")
b3 = utils.bias_variable([64], name="b3")
h3 = utils.conv2d_transpose(h2, W3, b3, output_shape_h3, strides=[1, 2, 2, 1], padding="SAME")
```




# NLP application
* Look for embedding in a matrix given an id:
```
tf.nn.embedding_lookup(embeddings, mat_ids)
# where embeddings is a tf.Variable()
```

## RNN, LSTM, and shits
LSTM works better than RNN to remenber long term dependencies, because in its form, only the forget gate get multiplied over time, and not activation, hence as long as the forget bias _ft_ is close to 1, it backpropagate up to the time t. More than that, you must make sure to init the bias different than 0!
### Dynamic or static rnn
* Just use ```tf.dynamic_rnn```, it uses a ```tf.While``` allowing to dynamically construct the graph, and passing different sentence lengths between batches.

### Set state for LSTM cell stacked
1. A LSTM cell state contains two tensor (the context, and the hidden state). Let's create a placeholder for both this tensors  
    ```

    # create a (context tensor, hidden tensor) for every layers
    state_placeholder = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])
    # unpack them
    l = tf.unstack(state_placeholder, axis=0)
    ```
2. Transform them into tuples
    ```
    rnn_tuple_state = tuple(
             [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],l[idx][1])
              for idx in range(num_layers)]
    )
    ```
3. Create the dynamic rnn, and passed initialized state
    ```
    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    outputs, state = tf.nn.dynamic_rnn(cell, series_batch_input, initial_state=rnn_tuple_state)
    ```

### Stacking recurrent neural network cells
1. Create the architecture (example for a GRUCell with dropout between every stacking cell
    ```
    from tensorflow.nn.rnn_cell import GRUCell, DropoutWrapper, MultiRNNCell

    num_neurons = 200
    num_layers = 3
    dropout = tf.placeholder(0.1, tf.float32)

    cell = GRUCell(num_neurons)
    cell = DropoutWrapper(cell, output_keep_prob=dropout)
    cell = MultiRNNCell([cell] * num_layers)
    ```

2. Simulate the recurrent network over the time step of the input with ```dynamic_rnn```:
    ```
    output, state = tf.nn.dynamic_rnn(cell, some_variable, dtype=tf.float32)
    ```

### Variable sequence length input
* First of all ```dynamic_rnn()``` return an output vector, which is of size ```max_length_sentence x hidden_vector```. It contains all the hidden state at every timestep. The other return object is a state object. It is the current state of the cell (hence it depends of the cell used, lstm or rnn...).   
Often, passing sentences to RNN, not all of them are of the same length. Tensorflow wants us to pass into a RNN a tensor of shape ```batch_size x sentence_length x embedding_length```.
To support this in our RNN, we have to first create an 3D array where for each rows (every batch element), we pad with zeros after reaching the end of the batch element sentence. For example if the length of the first sentence is 10, and ```sentence_length=20```, then all element ```tensor[0,10:, :] = 0``` will be zero padded.  

1. It is possible to compute the length of every batch element with this function:  
    ```

    def length(sequence):
    	@sequence: 3D tensor of shape (batch_size, sequence_length, embedding_size)
    	used = tf.sign(tf.reduce_sum(tf.abs(sequence), reduction_indices=2))
    	length = tf.reduce_sum(used, reduction_indics=1)
    	length = tf.cast(length, tf.int32)
    	return length # vector of size (batch_size) containing sentence lengths
    ```
2. Using the length function, we can create our rnn  
    ```

    from tensorflow.nn.rnn_cell import GRUCell

    max_length = 100
    embedding_size = 32
    num_hidden = 120

    sequence = tf.placeholder([None, max_length, embedding_size])
    output, state = tf.nn.dynamic_rnn(
        GRUCell(num_hidden),
        sequence,
        dtype=tf.float32,
        sequence_length=length(sequence),
    )

    ```

There are two cases, whether we are interested by only the last element outputted, or all output at every timestep. Let's define a function for both of them

#### Case 1: output at each timesteps
__Example__: Compute the cross-entropy for every batch element of different size (we can't use ```reduce_mean()```)

```
targets = tf.placeholder([batch_size, sequence_length, output_size])
# targets is padded with zeros in the same way as sequence has been done
def cost(targets):
	cross_entropy = targets * tf.log(output)
	cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
	mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
	cross_entropy *= mask

	# Average over all sequence_length
	cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
	cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
	return tf.reduce_mean(cross_entropy)
```

#### Case 2: output at the last timestep
__Example__: Get the last output for every batch element:
```
def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
```

### Bidirectionnal Recurrent Neural Network
Not so different from the standart ```dynamic_rnn```, we just need to pass cell for forward and backward pass, and it will return two outputs, and two states variables.  d
Example:
```
cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
 
outputs, states  = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=cell, # same cell for both passes
    cell_bw=cell,
    dtype=tf.float64,
    sequence_length=X_lengths, # didn't mention them in the snippet
    inputs=X)
output_fw, output_bw = outputs
states_fw, states_bw = states
```


# Higher order operators
* tf.map_fn() : apply a function to a list of elements.
```
array = (np.array([1, 2]), np.array([2, 3])
tf.map_fn(lambda x: (x[0] + x[1], x[0] * x[1]), array)
# => return ((3, 5), (2, 6))
```
* tf.foldl(): accumulate and apply a function on a sequence.
```
array = np.array([1, 3, 4, 3, 2, 4])
tf.foldl(lambda a, x: a + x, array) => 17
tf.foldl(lambda a, x: a + x, array, initializer=3) => 20
tf.foldr(lamnbda a, x: a + x,  array) => -9
```
* tf.scan(): 
```
tf.scan(loop_element, range_element: function(), elems = all_elems_to_iterate_over,
						 initializer=  initializer
# function() should return a tensor of shape initializer
# loop_element is of shape initializer
# range_element is iterate over and is not always necessary (ex: np.arange(10))
# scan return a vector of all vector of shape initializer
```

* tf.while_loop(condition, body, init)
```
init = (i, (j,k))
condition = lambda i, _: i<10
body = lambda i, jk: return (i+1, (jk[0] - jk[1], jk[0] + jk[1]))
(i_final, jk_final) = tf.while_loop(condition, body, init)
```

# Debugging
It is usefull to be able to monitor the training of a neural network, and add filter for events, such as the apparition of nan or inf number. To do so, clip a tensor filter to the current session  
```
from tensorflow.python import debug as tf_debug

sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

# where the filter is defined this way
def has_inf_or_nan(datum, tensor):
  return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
```

Then, run the model ```python model.py --debug```. A commmand line will prompt at first sess.run.
Here is a non exhaustive list of usefull command:
* Page down\up to move in the page (clicking in the terminal is also available)
* Print the value of a tensor: ````pt hidden/Relu:0```
* Print a sub-array ```pt hidden/Relu:0[0:50,:]```
* Print a sub-array and highlight specific element in a given range ```pt hidden/Relu:0[0:10,:] -r [1,inf]```
* Navigate to the current index of a tensor being displayed @[10, 0]
* Search for regex pattern such as /inf or /nan
* Display information about the node attribute ```ni -a hidden/Relu:0```.
* Display information about the current run ```run_info or ri```
* ```help``` command
* Run a session until a filter catch something ```run -f filter_name``` (Note that the filter name is the filter name passed to add_tensor_filter).
* Run a session for a number of step: run -t 10

# Miscellanous
* ```tf.sign(var)``` return -1, 0, or 1 depending the var sign.
* ```tf.reduce_max(3D_tensor, reduction_indices=2)``` return a 2D tensor, where only the max element in the 3dim is kept.
* ```tf.unstack(value, axis=0)```: If given an array of shape (A, B, C, D), and an axis=2, it will return a list of |C| tensor of shape (A, B, D).
* ```tf.nn.moments(x, axes)```: return the mean and variance of the vector in the dimension=axis
* ```tf.nn.xw_plus_b(x, w, b)```: explicit
* tf.global_variables(): return every new variables that are shred across machines in a distributed environment. Each time a Variable() constructor is called, it adds a new variabl ot he graph collection
* tf.convert_to_tensor(args, dtype): (tf.convert_to_tensor([[1, 2],[2, 3]], dtype=tf.float32)): convert an numpy array, a python list or scalar, to a Tensor.
* ```tf.placeholder_with_default(defautl_output, shape)```: One can see a placeholder as an element in the graph that must be fed an output value with the feed dictionnary, however it is possible to define placeholder that take default value.
* ```tf.variable_scope(name_or_scope, default_name)```: if name_or_scope is None, then scope.name is default_name.
* ```tf.get_default_graph().get_operations()```: return all operations in the graph, operations can be filtered by scope then with the python function ```startwith```. It returns a list of tf.ops.Operation
* ```tf.expand_dims([1, 2], axis=1)``` return a tensor where the axis dimensions is expanded. Here the new shape will be (2) -> (2, 1)
* ```tf.groups(op_1, op_2, op_3)``` can be pass to sess.run and it will run all operations (but it will not return any output, only computed operations) 
* ```tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits) expects labels to be int32 of size (batchsize), where every element is an integer from 0 to nbclasses. logits should be a float32 vector of size (batchsize, nbclasses) with values in it are not probabilities (logit form, before softmax)
* FLAGS is an internal mecanism that allowed the same functionnality as argparse
* ```clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_value_min, clip_value_max)) for
                                         var in list_tf_variables]``` create an operator to run in a sess that will clip values.


# Tensorflow fold
All tensorflow_fold function to treat sequences:
* td.Map(f): Takes a sequence as input, applies block f to every element in the sequence, and produces a sequence as output.
* td.Fold(f, z): Takes a sequence as input, and performs a left-fold, using the output of block z as the initial element.
* td.RNN(c): A recurrent neural network, which is a combination of Map and Fold. Takes an initial state and input sequence, uses the rnn-cell c to produce new states and outputs from previous states and inputs, and returns a final state and output sequence.
* td.Reduce(f): Takes a sequence as input, and reduces it to a single value by applying f to elements pair-wise, essentially executing a binary expression tree with f.
* td.Zip(): Takes a tuple of sequences as inputs, and produces a sequence of tuples as output.
* td.Broadcast(a): Takes the output of block a, and turns it into an infinite repeating sequence. Typically used in conjunction with Zip and Map, to process each element of a sequence with a function that uses a.

# Tensorflow Estimator
### Create an estimator
```python
nn = tf.contrib.learn.Estimator(model_fn = a_function,
								params=some_parameters,
								model_dir=where_to_log,
								contrib=tf.contrib.learn.RunConfig(save_checkpoints_sec=10))
```    

### The model function
Here is a simple example of model function
```
def model_fn(features, targets, mode, params):
	"""
	features: vector of feature vector
	targets: vector of labels
	model: string (train or test)
	params: a dictionnary of params (optionnal)
	"""

	# 1. get input
	# 2. build NN
	# 3. output = get_output
	prediction = create_vector_of_prediction_output()
	loss = create_loss_function(targets, output)
	accuracy = accuracy_function(targets, output)

	# You can had any summary in this function, they will be saved when building the model

	# Call optimize_loss
	train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=some_lr_rate_values,
        optimizer=(lambda l_r: optimizer(l_r)),
        summaries=tf.GraphKeys.TRAIN_OP)

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=prediction,
        loss=loss,
        train_op=train_op)  

```


### Monitor some values
You can monitor specific tensor, or also some metrics. First you need to create some ValidationMonitor function:  
#### Monitor early stopping or some accuracy
```
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        x=x_val,
        y=y_val,
        metrics=metric_function
        early_stopping_metric="loss", # early stopp on xval and yval given some metrics         
        early_stopping_metric_minimize=True,
        early_stopping_rounds=int)
```  
The metric function is of form
```
def metric_function(predictions, labels):
    return value
```
### Monitor values of tensor
```
tensors_to_log = {"Random name": "name_of_tensor_in_graph"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=print_each_nb_iters)
```
### Fit a model
```
nn.fit(x=x_train, y=y_train, steps=200, monitors=[logging_hook, validation_monitor])
```

### Evaluate a model
```
nn.evaluate(x=x_test, y=y_test)
```