# Import the relevant packages
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

# Data

mnist_dataset, mnist_info = tfds.load('mnist', with_info=True, as_supervised=True)
# tfds.load(name) loads a dataset from TensorFlow datasets
#-> as_supervised = True, loads the data in a 2-tuple structure [input, target]
#-> with_info = True, provides a tuple containing info about version, features, # samples of dataset

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)
# tf.cast(x, dtype) casts (converts) a variable into a given data type

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label
# dataset.map(*function*) applies a custom transformation to a given dataset. It takes as input a function which
#determines the transformation

scaled_train_and_validation_data = mnist_train.map(scale)

test_data = mnist_test.map(scale)

BUFFER_SIZE = 10000
# Note: If buffer_size = 1, no shuffeling will actually happen if buffer_size >= num_samples, shuffling happend at once

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)
# batch size = 1 = Stochastic gradient descent (SGD) batch size = # samples = (single batch) GD

BATCH_SIZE = 100

# dataset.batch(batch_size) a method that combines the consecutive elements of a dataset into batches

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

validation_inputs, validation_targets = next(iter(validation_data))
# next() loads the next element of an iterable object
# iter() creates an object which can be iterated one element at a time (e.g. in a for loop or while loop)

# Model

# Outline the model
input_size = 784
output_size = 10
hidden_layer_size = 100

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')

])
# tf.keras.Sequential() unction that is laying down the model (used to 'stack layers')
# tf.keras.layers.Flatten(original shape) transforms (flattens) a tensor into a vector
# tf.keras.layers.Dense(output size) takes the inputs, provided to the model and calculates the dot product
#of the inputs and the wieghts and adds the bias. This is also where we can apply an activation function

# Choose the optimizer and the loss functon

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer, loss, metrics)

NUM_EPOCHS = 5

model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose=2)

# Test the model
# model.evaluate() returns the loss value and metrics values for the model in 'test mode'

test_loss, test_accuracy = model.evaluate(test_data)
print('Test loss: {0:.2f}. Test accuracy {1:.2f}'.format(test_loss, test_accuracy*100.))
