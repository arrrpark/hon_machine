from tensorflow import keras
import numpy as np

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()


import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1, 10, figsize=(10,10))
# for i in range(10):
#     axs[i].imshow(train_input[i], cmap='gray_r')
#     # axs[i].axis['off']

# plt.show()

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28 * 28)
print(train_scaled.shape)

from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)

model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
    keras.layers.Dense(10, activation='softmax', name='output')
])

model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs = 10, verbose=0, validation_data=(val_scaled, val_target))
model.evaluate(val_scaled, val_target)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
