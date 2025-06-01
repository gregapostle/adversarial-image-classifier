import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import numpy as np
import random
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Model
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Generate adversarial examples
x_test_adv = fast_gradient_method(model_fn=model,
                                  x=x_test,
                                  eps=0.3,
                                  norm=np.inf,
                                  targeted=False)

# Evaluate accuracy
loss, acc = model.evaluate(x_test_adv, y_test)
print(f'Accuracy on adversarial examples: {acc:.2%}')

# How many examples to show
n = 5

# Randomly select n indices
indices = random.sample(range(len(x_test)), n)

plt.figure(figsize=(10, 2 * n))

for idx, i in enumerate(indices):
    # Get predictions
    true_label = class_names[y_test[i]]
    pred_orig = class_names[tf.argmax(model.predict(x_test[i:i+1]), axis=1).numpy()[0]]
    pred_adv = class_names[tf.argmax(model.predict(x_test_adv[i:i+1]), axis=1).numpy()[0]]


    # Original image
    plt.subplot(n, 2, 2*idx + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Original\nTrue: {true_label} | Pred: {pred_orig}")
    plt.axis('off')

    # Adversarial image
    plt.subplot(n, 2, 2*idx + 2)
    plt.imshow(x_test_adv[i], cmap='gray')
    plt.title(f"Adversarial\nPred: {pred_adv}")
    plt.axis('off')

plt.tight_layout()
plt.show()