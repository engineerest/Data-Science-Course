# Import the relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate random input data to train on
observations = 1000
# np.random.uniform(low, hight, size) draws a random value from the interval (low, high), where each number has as
# equal chance to be selected

xs = np.random.uniform(low=-10,high=10,size=(observations,1))
zs = np.random.uniform(-10,10,(observations,1))

inputs = np.column_stack((xs,zs))

print(inputs.shape)

# Create the target we will aim at
noise = np.random.uniform(-1,1,(observations,1))

targets = 2*xs - 3*zs + 5 + noise

print(targets.shape)

# Plot the training data

targets = targets.reshape(observations,)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, zs, targets)
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('targets')
ax.view_init(azim=100)
plt.show()
targets = targets.reshape(observations,1)

# Intialize variables
init_range = 0.1

weights = np.random.uniform(-init_range,init_range,(2,1))

biases = np.random.uniform(-init_range,init_range,1)

print(weights)
print(biases)

# Set a learning rate
learning_rate = 0.02

# Train the model
for i in range(100):
    outputs = np.dot(inputs, weights) + biases
    deltas = outputs - targets
    # np.sum(a) is a method that allows us to sum all the values in the array
    loss = np.sum(deltas ** 2) / 2 / observations

    print(loss)

    deltas_scaled = deltas / observations

    weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled)
    biases = biases - learning_rate * np.sum(deltas_scaled)

# Print weights and biases and see if we have worked correctly
print(weights, biases)

plt.plot(outputs, targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()