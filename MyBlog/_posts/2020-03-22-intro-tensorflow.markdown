---
layout: post
title:  "Introduction to TensorFlow"
date:   2020-12-22 09:15:16 +0200
categories: machine learning
---
TensorFlow is a widely used machine learning library. Tensors are n-dimensional arrays of data and the TensorFlow library helps define computation graphs with them that can be submitted for execution to local CPUs, GPUs or remote clusters for big data processing. 

### Hello World

Before starting, let's make sure we run the latest version of TensorFlow 2.x

```
$pip install --upgrade tensorflow
$ipython
In [1]: import tensorflow as tf
In [2]: tf.__version__
Out[2]: '2.1.0'
```

Unlike previous version, by default TensorFlow has enabled eager execution. Therefore, we can evaluate directly code like the following, without creating a `Session` object.

```python
import tensorflow as tf

s = tf.constant("Hello world")
a = tf.constant(1)
b = tf.constant(2)

# .numpy() is needed to get data back from the graph to the current process
print(s.numpy())
print((a+b).numpy())
```

### A More Complex Hello World

In the example below we will train a multiclass classifier based on the Fashion Minst dataset using tensor flow. We will output the F1 score for each class, which gives pretty good results.

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist

(timg, tlbl), (tstimg, tstlbl) = fashion_mnist.load_data(); 

# normalize from 0..255 to 0..1
timg = timg / 255
tstimg = tstimg / 255

plt.imshow(timg[2])
print(tlbl[1])


# regular 3 layers feed-forward neural network
# 10 output neurons, corresponding to each of the classes
model = keras.Sequential([
    
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu', use_bias=True),
    keras.layers.Dense(10)
    
    ])

# multi-class
# https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
   
model.compile(
    optimizer='adam', 
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
    )

model.fit(timg, tlbl, epochs=10)

# predict -> we are interested in the index of the neuron which has the highest value
# out of the 10 output neurons
predictions = np.argmax(model.predict(tstimg), axis=1)

from sklearn.metrics import f1_score
print(f1_score(tstlbl, predictions, average=None))
```
