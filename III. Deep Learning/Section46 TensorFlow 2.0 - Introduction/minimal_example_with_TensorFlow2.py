# Import the relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Data generation
observations = 100
xs = np.random.uniform(-10, 10, (observations, 1))
zs = np.random.uniform(-10, 10, (observations, 1))

generated_inputs = np.column_stack((xs, zs))

noise = np.random.uniform(-1, 1, (observations, 1))

generated_targets = 2*xs - 3*zs + 5 + noise

np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)
# np.savez(file name, arrays) saves n-dimensional arrays in .npz format, using
#a certain keyword (label) for each array

# Solving with TensorFlow
training_data = np.load('TF_intro.npz')
input_size = 2
output_size = 1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(output_size,
                          kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                          bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),)
])
# tf.keras.Sequential() function that specifies how the model will be laid down ('stacks layers')
# tf.keras.layers.Dense(output size) takes the inputs provided to the model and calculates the dot
#product of the inputs and the weights and adds the bias * also applies activation function (optional)
# tf.keras.layers.Dense(output_size, kernel_initializer, bias_initializer) function that is laying down
#the model (used to stack layers') adn intialize weights

custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
# tf.keras.optimizers.SGD(learning_rate) Stochastic gradient descent optimizers, including
#support for learning rate, momentum, decay, etc.

model.compile(optimizer=custom_optimizer, loss="mean_squared_error")
# model.compile(optimizer, loss) configures the model for training
model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=2) # Epoch = iteration over the full dataset
# model.fit(inputs, targets) fits (trains) the model


# Extract the weights and bias
model.layers[0].get_weights()
weights = model.layers[0].get_weights()[0]
print(weights)
bias = model.layers[0].get_weights()[1]
print(bias)

# Extract the outputs (make predictions)

print(model.predict_on_batch(training_data['inputs']).round(1))
print(training_data['targets'].round(1))
# model.predict_on_batch(data) calculates the outputs given inputs

# Plotting the data
plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()