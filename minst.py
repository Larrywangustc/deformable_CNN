from __future__ import absolute_import, division
# %env CUDA_VISIBLE_DEVICES=0
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
from Layers import ConvOffset2D
from callbacks import TensorBoard
from minst_model import get_cnn_sample, get_deform_cnn_sample
from keras.layers import Conv2D
import keras
import os
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_mnist_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = X_train[..., None]
    X_test = X_test[..., None]
    Y_train = tf.keras.utils.to_categorical(y_train, 10)
    Y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (X_train, Y_train), (X_test, Y_test)


def get_gen(set_name, batch_size, translate, scale, rot=0, flip=0,
            shuffle=True):
    if set_name == 'train':
        (X, Y), _ = get_mnist_dataset()
    elif set_name == 'test':
        _, (X, Y) = get_mnist_dataset()

    image_gen = ImageDataGenerator(
        zoom_range=scale,
        width_shift_range=translate,
        height_shift_range=translate,
        rotation_range=rot,
        vertical_flip=flip
    )
    gen = image_gen.flow(X, Y, batch_size=batch_size, shuffle=shuffle)
    return gen

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

# ---
# Config

batch_size = 32
n_train = 60000
n_test = 10000
steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = int(np.ceil(n_test / batch_size))

train_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=True
)
test_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=False
)
train_scaled_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=True, rot=70, flip=1
)
test_scaled_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=False, rot=70, flip=1
)



Xb, Yb = next(test_gen)
print(np.shape(Xb))
Xb = Xb.reshape(-1, 28, 28)
plt.figure()
x = np.linspace(0, 1, 28)
y = np.linspace(0, 1, 28)
X, Y = np.meshgrid(x, y)
plt.figure()
plt.imshow(Xb[11], cmap=plt.get_cmap('gray'))
plt.show()
a=1






# ---
# Normal CNN

inputs, outputs = get_cnn_sample()
model = Model(inputs=inputs, outputs=outputs)
model.summary()
optim = Adam(1e-3)
# optim = SGD(1e-3, momentum=0.99, nesterov=True)
loss = categorical_crossentropy
model.compile(optim, loss, metrics=['accuracy'])

model.fit_generator(
    train_gen, steps_per_epoch=steps_per_epoch,
    epochs=10, verbose=1,
    validation_data=test_gen, validation_steps=validation_steps
)
model.save_weights('models/cnn.h5')
# 1875/1875 [==============================] - 24s - loss: 0.0090 - acc: 0.9969 - val_loss: 0.0528 - val_acc: 0.9858

# ---
# Evaluate normal CNN

model.load_weights('models/cnn.h5', by_name=True)

val_loss, val_acc = model.evaluate(
    test_gen, steps=validation_steps
)
print('Test accuracy', val_acc)
# 0.9874

val_loss, val_acc = model.evaluate(
    test_scaled_gen, steps=validation_steps
)
print('Test accuracy with scaled images', val_acc)
# 0.5701

# Deformable CNN

inputs, outputs = get_deform_cnn_sample(trainable=False)
model = Model(inputs=inputs, outputs=outputs)
model.load_weights('models/cnn.h5', by_name=True)
model.summary()
optim = Adam(5e-4)
# optim = SGD(1e-4, momentum=0.99, nesterov=True)
loss = categorical_crossentropy
model.compile(optim, loss, metrics=['accuracy'])

model.fit_generator(
    train_scaled_gen, steps_per_epoch=steps_per_epoch,
    epochs=10, verbose=1,
    validation_data=test_scaled_gen, validation_steps=validation_steps
)
# Epoch 20/20
# 1875/1875 [==============================] - 504s - loss: 0.2838 - acc: 0.9122 - val_loss: 0.2359 - val_acc: 0.9231
model.save_weights('models/deform_cnn.h5')

# --
# Evaluate deformable CNN

model.load_weights('models/deform_cnn.h5')

val_loss, val_acc = model.evaluate(
    test_scaled_gen, steps=validation_steps
)
print('Test accuracy of deformable convolution with scaled images', val_acc)
# 0.9255

val_loss, val_acc = model.evaluate(
    test_gen, steps=validation_steps
)
print('Test accuracy of deformable convolution with regular images', val_acc)
# 0.9727'''

deform_conv_layers = [l for l in model.layers if isinstance(l, ConvOffset2D)]

Xb, Yb = next(test_gen)
for l in deform_conv_layers:
    print(l)
    _model = Model(inputs=inputs, outputs=l.output)
    offsets = _model.predict(Xb)
    offsets = offsets.reshape(offsets.shape[0], offsets.shape[1], offsets.shape[2], -1, 2)
    print(offsets.min())
    print(offsets.mean())
    print(offsets.max())
    print(np.shape(l.weights[-1].numpy()))
    print(np.shape(l.weights[-1].numpy()))

conv_layers = [l for l in model.layers if isinstance(l, Conv2D)]

Xb, Yb = next(test_gen)
for l in conv_layers:
    _model = Model(inputs=inputs, outputs=l.output)
    offsets = _model.predict(Xb)
    offsets = offsets.reshape(offsets.shape[0], offsets.shape[1], offsets.shape[2], -1, 2)
    print(offsets.min())
    print(offsets.mean())
    print(offsets.max())
