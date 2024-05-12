import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X = housing.data
y_task1 = housing.target.reshape(-1, 1)  
y_task2 = X[:, -1].reshape(-1, 1)  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_task1_train, y_task1_test, y_task2_train, y_task2_test = train_test_split(
    X_scaled, y_task1, y_task2, test_size=0.2, random_state=42)


input_layer = tf.keras.layers.Input(shape=(X_train.shape[1],))
shared_layer = tf.keras.layers.Dense(10, activation='relu')(input_layer)
output_task1 = tf.keras.layers.Dense(1, name='task1')(shared_layer)
output_task2 = tf.keras.layers.Dense(1, name='task2')(shared_layer)

model = tf.keras.models.Model(inputs=input_layer, outputs=[output_task1, output_task2])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(X_train, [y_task1_train, y_task2_train], epochs=100, validation_split=0.2, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])

plt.plot(history.history['loss'], label='Task 1 Training Loss')
plt.plot(history.history['val_loss'], label='Task 1 Validation Loss')
plt.plot(history.history['task2_loss'], label='Task 2 Training Loss')
plt.plot(history.history['val_task2_loss'], label='Task 2 Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Multi-Task Learning with Early Stopping')
plt.legend()

