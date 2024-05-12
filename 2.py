import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=32, n_classes=2, random_state=42)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define deep neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with Adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with Adam optimizer (simulating Gradient Descent)
gd_history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

# Compile the model with Stochastic Gradient Descent optimizer
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with Stochastic Gradient Descent optimizer
sgd_history = model.fit(X_train, y_train, epochs=20, batch_size=1, validation_split=0.2, verbose=0)

plt.plot(gd_history.history['loss'], label='GD Training Loss')
plt.plot(gd_history.history['val_loss'], label='GD Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(sgd_history.history['loss'], label='SGD Training Loss')
plt.plot(sgd_history.history['val_loss'], label='SGD Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
