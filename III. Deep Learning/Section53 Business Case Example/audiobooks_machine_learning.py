# Import the relevant libraries

import numpy as np
import tensorflow as tf

# Data
npz = np.load('Audiobooks_data_train.npz')

train_inputs = npz['inputs'].astype(np.float_)
# np.ndarray.astype() creates a copy of the array, cst to a specific type
train_targets = npz['targets'].astype(np.int_)

npz = np.load('Audiobooks_data_validation.npz')
validation_inputs, validation_targets = npz['inputs'].astype(np.float_), npz['targets'].astype(np.int_)

npz = np.load('Audiobooks_data_test.npz')
test_inputs, test_targets = npz['inputs'].astype(np.float_), npz['targets'].astype(np.int_)

# Model
input_size = 784
output_size = 10
hidden_layer_size = 50

model = tf.keras.Sequential([
    # tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 100

max_epochs = 100

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)
# tf.keras.callbacks.EarlyStopping(patience) configures the early stopping mechanism of the algorithm.
#'patience' lets us decide how many consecutive increases we can tolerate

model.fit(train_inputs, train_targets, batch_size=batch_size, epochs=max_epochs, callbacks=[early_stopping], validation_data=(validation_inputs, validation_targets), verbose=2)

test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print('\n Test loss: {0:.2f}. Test accuracy: {1:.2f}'.format(test_loss, test_accuracy*100.))

