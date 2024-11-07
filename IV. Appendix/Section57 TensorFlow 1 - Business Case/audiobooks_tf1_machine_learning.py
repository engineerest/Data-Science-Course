# Crate a class that will batch the data

import numpy as np


# Create a class that will do the batching for the algorithm
# This code is extremely reusable. You should just change Audiobooks_data everywhere in the code
class Audiobooks_Data_Reader():
    # Dataset is a mandatory arugment, while the batch_size is optional
    # If you don't input batch_size, it will automatically take the value: None
    def __init__(self, dataset, batch_size=None):

        # The dataset that loads is one of "train", "validation", "test".
        # e.g. if I call this class with x('train',5), it will load 'Audiobooks_data_train.npz' with a batch size of 5.
        npz = np.load('Audiobooks_data_{0}.npz'.format(dataset))

        # Two variables that take the values of the inputs and the targets. Inputs are floats, targets are integers
        self.inputs, self.targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

        # Counts the batch number, given the size you feed it later
        # If the batch size is None, we are either validating or testing, so we want to take the data in a single batch
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size

    # A method which loads the next batch
    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()

        # You slice the dataset in batches and then the "next" function loads them one after the other
        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1

        # One-hot encode the targets. In this example it's a bit superfluous since we have a 0/1 column
        # as a target already but we're giving you the code regardless, as it will be useful for any
        # classification task with more than one target column
        classes_num = 2
        targets_one_hot = np.zeros((targets_batch.shape[0], classes_num))
        targets_one_hot[range(targets_batch.shape[0]), targets_batch] = 1

        # The function will return the inputs batch and the one-hot encoded targets
        return inputs_batch, targets_one_hot

    # A method needed for iterating over the batches, as we will put them in a loop
    # This tells Python that the class we're defining is iterable, i.e. that we can use it like:
    # for input, output in data:
    # do things
    # An iterator in Python is a class with a method __next__ that defines exactly how to iterate through its objects
    def __iter__(self):
        return self

# Create the machine learning algorithm

import tensorflow as tf

input_size = 10
output_size = 2
hidden_layer_size = 100

tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.int32, [None, output_size])

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

max_epochs = 50

prev_validation_loss = 9999999.

train_data = Audiobooks_Data_Reader('train', batch_size)
validation_data = Audiobooks_Data_Reader('validation')

for epoch_counter in range(max_epochs):

    curr_epoch_loss = 0.

    for input_batch, target_batch in train_data:
        _, batch_loss = sess.run([optimize, mean_loss],
        feed_dict={inputs: input_batch, targets: target_batch})

        curr_epoch_loss += batch_loss

    curr_epoch_loss /= train_data.batch_count

    validation_loss = 0.
    validation_accuracy = 0.

    for input_batch, target_batch in validation_data:
        validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
        feed_dict={inputs: input_batch, targets: target_batch})

    print('Epoch '+str(epoch_counter+1)+
          '. Training loss: '+'{0:.3f}'.format(curr_epoch_loss)+
          '. Validation loss: '+'{0:.3f}'.format(validation_loss)+
          '. Validation accuracy: '+'{0:.2f}'.format(validation_accuracy * 100.)+'%')

    if validation_loss > prev_validation_loss:
        break

    prev_validation_loss = validation_loss

print('End of training.')

# Test the model

test_data = Audiobooks_Data_Reader('test')

for input_batch, target_batch in test_data:
    test_accuracy = sess.run(accuracy, feed_dict={inputs: input_batch, targets: target_batch})

test_accuracy_percent = test_accuracy[0] * 100

print('Test accuracy: '+'{0:.2f}'.format(test_accuracy_percent)+'%')