import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, _), (_, _) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32')
X_train = MinMaxScaler().fit_transform(X_train)

rbm_layers = [64, 64, 784]

input_data = X_train
rbm_models = []

for n_components in rbm_layers:
    rbm = BernoulliRBM(n_components=n_components, learning_rate=0.1, n_iter=10, random_state=42)
    rbm.fit(input_data)
    rbm_models.append(rbm)
    input_data = rbm.transform(input_data)

print("Final representation shape:", input_data.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(X_train[0].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
axes[0].set_title('Original Image')
axes[1].imshow(input_data[0].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
axes[1].set_title('Transformed Image')

plt.show()
