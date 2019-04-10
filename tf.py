import tensorflow as tf
from tensorflow import keras

import numpy as np
import  matplotlib.pyplot as plt


def test():
    print(tf.__version__)
    fashion_mnist = keras.datasets.fashion_mnist
    (train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(train_img.shape)
    print("x:" + str(len(train_labels)))
    print(train_labels)

    train_img = train_img / 255.0
    test_img = test_img / 255.0
    fig = plt.figure(figsize=(10, 10))
    '''
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_img[i],cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    '''
    model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                              keras.layers.Dense(128, activation=tf.nn.relu),
                              keras.layers.Dense(10, activation=tf.nn.softmax)])

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_img, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_img, test_labels)
    print('test acc:', test_acc)

    perdiction = model.predict(test_img)
    print(perdiction[0])
    x = np.argmax(perdiction[0])
    print('perd:', x, test_labels[0])
    i = 0
    plt.figure(figsize=(6, 3))
    




