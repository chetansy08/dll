import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# Adversarial Training
def adversarial_training(model, X_train, y_train, epsilon=0.01, epochs=10):
    history = {'loss': [], 'accuracy': []}
    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        for i in range(len(X_train)):
            x = X_train[i:i+1]
            y = y_train[i:i+1]
            with tf.GradientTape() as tape:
                preds = model(x)
                loss = tf.keras.losses.categorical_crossentropy(y, preds)
            gradients = tape.gradient(loss, model.trainable_variables)
            perturbations = [epsilon * tf.sign(grad) for grad in gradients]
            for j in range(len(perturbations)):
                model.trainable_variables[j].assign_add(perturbations[j])
            epoch_loss += loss.numpy().mean()
            correct_predictions += np.argmax(preds) == np.argmax(y)
        history['loss'].append(epoch_loss / len(X_train))
        history['accuracy'].append(correct_predictions / len(X_train))
    return history

# Tangent Distance
def tangent_distance(x1, x2):
    return euclidean_distances([x1.ravel()], [x2.ravel()])[0][0]

# Tangent Propagation
def tangent_propagation(model, X_train, y_train, epochs=10):
    history = {'loss': [], 'accuracy': []}
    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        for i in range(len(X_train)):
            x = X_train[i:i+1]
            y = y_train[i:i+1]
            with tf.GradientTape() as tape:
                preds = model(x)
                loss = tf.keras.losses.categorical_crossentropy(y, preds)
            gradients = tape.gradient(loss, model.trainable_variables)
            for j in range(len(gradients)):
                model.trainable_variables[j].assign_add(-gradients[j])
            epoch_loss += loss.numpy().mean()
            correct_predictions += np.argmax(preds) == np.argmax(y)
        history['loss'].append(epoch_loss / len(X_train))
        history['accuracy'].append(correct_predictions / len(X_train))
    return history

# def tangent_classifier(X_train, y_train, X_test):
#     predictions = []
#     for test_sample in X_test:
#         distances = [tangent_distance(test_sample, train_sample) for train_sample in X_train]
#         nearest_index = np.argmin(distances)
#         predictions.append(y_train[nearest_index])
#     return predictions

def plot_history(history, title):

    plt.plot(history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} Loss')
    plt.legend()
    plt.show()

X_train = np.random.rand(100, 784)
y_train = np.random.randint(0, 2, size=(100, 10))
X_test = np.random.rand(20, 784)

model = Sequential()
model.add(Dense(10, input_shape=(784,), activation='relu'))
model.add(Dense(10, activation='softmax'))

adv_history = adversarial_training(model, X_train, y_train)
plot_history(adv_history, 'Adversarial Training')

tangent_history = tangent_propagation(model, X_train, y_train)
plot_history(tangent_history, 'Tangent Propagation')

predictions = tangent_classifier(X_train, y_train, X_test)
print(predictions)
