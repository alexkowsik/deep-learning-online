import math
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'Georgia'


# Code examples based on: https://towardsdatascience.com/intuition-behind-residual-neural-networks-fa5d2996b2c7


def sin():
    while True:
        x = tf.random.normal((30, 10), math.pi, 1.33)
        y = tf.math.sin(x)
        yield (x, y)

def resblock(inputs):
    x = tf.keras.layers.Dense(30, activation="relu")(inputs)
    x = tf.keras.layers.Dense(30)(x)
    x = x + inputs
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    return x

def plain_model(layers):
    x = inputs = tf.keras.Input((10, ))
    for _ in range(layers):
        x = tf.keras.layers.Dense(30, activation="relu")(x)
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, x)
    model.compile("adam", "mse")
    return model

def resnet_model(layers):
    x = inputs = tf.keras.Input((10, ))
    x = tf.keras.layers.Dense(30, activation="relu")(x)
    for _ in range(layers // 2 - 1):
        x = resblock(x)
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, x)
    model.compile("adam", "mse")
    return model

dts = tf.data.Dataset.from_generator(sin, (tf.float32, tf.float32), ((30, 10), (30, 10)))

errors_plain = []

for t in range(1, 30):
    plain = plain_model(t)
    hist = plain.fit(dts, steps_per_epoch=20, epochs=20*t, verbose=0)
    errors_plain.append(hist.history["loss"][-1])
    print(t, errors_plain[-1])

print(errors_plain)

plt.plot(errors_plain, color="#5086de")
plt.ylabel("Loss")
plt.xlabel("Anzahl der layer")
plt.title("Lernen der Sinus-Funktion mit einem einfachen Netz")
plt.savefig('plain_net.png')

errors_res = []

for t in range(1, 30):
    plain = resnet_model(t)
    hist = plain.fit(dts, steps_per_epoch=20, epochs=20*t, verbose=0)
    errors_res.append(hist.history["loss"][-1])
    print(t, errors_res[-1])

print(errors_res)

plt.plot(errors_plain, color="#5086de", label="Einfaches Netz")
plt.plot(errors_res, color="#ed9e2f", label="ResNet")
plt.ylabel("Loss")
plt.xlabel("Anzahl der layer")
plt.title("Sinus-Funktion: einfaches Netz vs ResNet")
plt.legend()
plt.savefig('res_net.png')
plt.show()
