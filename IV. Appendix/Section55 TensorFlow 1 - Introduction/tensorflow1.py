# Import the relevant libraries

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Data generation

observations = 1000

xs = np.random.uniform(-10, 10, (observations, 2))
zs = np.random.uniform(-10, 10, (observations, 1))

generated_inputs = np.column_stack((xs, zs))

noise = np.random.uniform(-1, 1, (observations, 1))
generated_targets = 2*xs - 3*zs + 5 + noise

np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)

# Solving with TensorFlow

input_size = 2
output_size = 1

# Outlining the model

inputs = tf.placeholder(tf.float32,[None, input_size])
targets = tf.placeholder(tf.float32,[None, output_size])

weights = tf.Variable(tf.random_uniform([input_size, output_size], minval=-0.1, maxval=0.1))
biases = tf.Variable(tf.random_uniform([output_size], minval=-0.1, maxval=0.1))

outputs = tf.matmul(inputs, weights) + biases
# tf.matmul(A, B, ...) relates to np.dot(A,B)

# Choosing the objective function and the optimization method

mean_loss = tf.losses.mean_squared_error(labels=targets, predictions=outputs) / 2.
# tf.losses is a module that contains most of the common loss functions

optimize = tf.train.GradientDescentOptimizer(0.05).minimize(mean_loss)
# tf.train is a module that contains most of the common optimization algos

# Prepare for execution

sess = tf.InteractiveSession()
# tf.InteractiveSession() is a TensorFlow class that is used whenever we want to execute something, anything

# Initializing variables
initializer = tf.global_variables_initializer()
# tf.global_variables_initializer() is a method that initializes all tensor objects "marked" as variables
# sess.run() is a method used for executing something, anything
sess.run(initializer)

# Loading training data
training_data = np.load('TF_intro.npz')

# Learning
for e in range(100):
    _, curr_loss = sess.run([optimize, mean_loss],
                            feed_dict = {inputs: training_data['inputs'], targets: training_data['targets']})
    print(curr_loss)

