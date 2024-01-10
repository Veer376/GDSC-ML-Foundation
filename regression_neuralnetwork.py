import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.random.set_seed(42)
np.random.seed(42)
x = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([65, 75, 80, 82, 88, 92, 95], dtype=float)

model = tf.keras.Sequential() # creating the stack of layers inside the model variable

model.add(tf.keras.layers.Dense(units=1, input_shape=[1])) # units is the number of neuron and input shape is the dimensionality of the input
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=1)

model.compile(optimizer=custom_optimizer, loss="mean_squared_error")

history = model.fit(x, y, epochs=500, verbose=0)

X = np.array([3.5, 4.5, 9.5], dtype=float)
predictions = model.predict(X)

for i in range(len(X)):
    print(f"For {X[i]} hours, predicted score: {predictions[i][0]}")


intercept = model.layers[0].get_weights()[1]
slope = model.layers[0].get_weights()[0]

print("Intercept:", intercept)
print("Slope:", slope)

plt.scatter(x,y,color="green")

# plt.plot(history.history['loss'])
# plt.title('Model Loss')
# plt.xlabel('Epochs')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,model.predict(x),color="red")
plt.show()
