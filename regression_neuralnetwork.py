import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Creating the dataset
x_train = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)
y_train = np.array([65, 75, 80, 82, 88, 92, 95], dtype=float)

# Creating a Sequential model
model = tf.keras.Sequential()

# Adding a single neuron Dense layer for linear regression
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
history = model.fit(x_train, y_train, epochs=500, verbose=0)

# Predicting exam scores for some study hours
x_test = np.array([3.5, 4.5, 9.5], dtype=float)
predictions = model.predict(x_test)

# Printing predicted scores
for i in range(len(x_test)):
    print(f"For {x_test[i]} hours, predicted score: {predictions[i][0]}")

# Plotting the training loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
