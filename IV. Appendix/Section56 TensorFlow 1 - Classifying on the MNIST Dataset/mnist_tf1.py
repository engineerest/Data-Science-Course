# Import the relevant packages

import numpy as np
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow_datasets as tfds

mnist = tfds.load("mnist") # Used TensorFlow 2

# Outline the model

input_size = 784
output_size = 10
hidden_layer_size = 50

tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])

# tf.get_variable("name", shape) is a function used to declare variables. The default initializer is Xavier (Glorot)
weights_1 = tf.get_variable("weights_1", [input_size, hidden_layer_size])
biases_1 = tf.get_variable("biases_1", [hidden_layer_size])

outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)
# tf.nn is a module that contains neural network (nn) support. Among other things,
#it contains the most commonly used activation
#functions

weights_2 = tf.get_variable("weights_2", [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable("biases_2", [hidden_layer_size])

outputs_2 = tf.nn.relu(tf.matmul(outputs_1, weights_2) + biases_2)

weights_3 = tf.get_variable("weights_3", [hidden_layer_size, output_size])
biases_3 = tf.get_variable("biases_3", [output_size])

outputs = tf.matmul(outputs_2, weights_3) + biases_3

loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
# tf.nn.softmax_cross_entropy_with_logits() is a functon that applies a softmax activation
#and calculates a cross-entropy loss

mean_loss = tf.reduce_mean(loss)
# tf.reduce_mean() is a method which finds the mean of the elements of a tensor across a dimension

optimize = tf.train.GradientDescentOptimizer(0.001).minimize(mean_loss)\

out_equals_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
# tf.equal() is a method that checks if two values are equal. In the case of tensor it does so element-wise

accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))
# tf.cast(object, data type) is a method that cast an object to another data type

sess = tf.InteractiveSession()

initializer = tf.global_variables_initializer()

sess.run(initializer)

batch_size = 100
# Batch size = 1 --> SGD
# Batch size = # samples --> GD

# We want a number small enough to learn faster, but big enough to preserve the underlying dependencies

batches_number = mnist.train.num_examples // batch_size

max_epochs = 15

prev_validation_loss = 9999999.

for epoch_counter in range(max_epochs):
    curr_epoch_loss = 0.
    for batch_counter in range(batches_number):
        input_batch, target_batch = mnist.train.next_batch(batch_size)
        # mnist.train.next_batch(size of the batch) is a function that comes with the MNIST data provider, which
        #loads the bathces one after the other

        _, batch_loss = sess.run([optimize, mean_loss],
                                 feed_dict={inputs: input_batch, targets: target_batch})

        curr_epoch_loss += batch_loss

    curr_epoch_loss /= batches_number

    input_batch, target_batch = mnist.validation.next_batch(mnist.validation.num_examples)

    validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
                                                    feed_dict={inputs: input_batch, targets: target_batch})

    print('Epoch '+str(epoch_counter+1)+
          '. Training loss:'+'{0:.3f}'.format(curr_epoch_loss)+
          '. Validation loss:'+'{0:.3f}'.format(validation_loss)+
          '. Validation accuracy:'+'{0:.2f}'.format(validation_accuracy * 100.)+'%')

    if validation_loss > prev_validation_loss:
        break

    prev_validation_loss = validation_loss

print('End of training.')


# Test

input_batch, target_batch = mnist.test.next_batch(mnist.test._num_examples)
test_accuracy = sess.run([accuracy],
                         feed_dict={inputs: input_batch, targets: target_batch})

test_accuracy_percent = test_accuracy[0] * 100.

print('Test accuracy:'+'{0:.2f}'.format(test_accuracy_percent)+'%')


