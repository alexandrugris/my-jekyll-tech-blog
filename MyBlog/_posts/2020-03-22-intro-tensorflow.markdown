---
layout: post
title:  "Introduction to TensorFlow"
date:   2021-12-22 09:15:16 +0200
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

In the example below we will train a multiclass classifier based on the Fashion Minst dataset using tensor flow. We will output the F1 score for each class, which gives pretty good results. We see that training takes quite a lot of data to be processed and takes a while. Predictions, on the other hand, are pretty fast. 

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

### Loading Data 

First example, load from CSV. Its main advantage is that, unlike pandas, it does not require reading the
full dataset in memory. You can read in batches and you can specify which columns to read. 

```python

TRAIN_URL = 'http://storage.googleapis.com/tf-datasets/titanic/train.csv'
TEST_URL = 'http://storage.googleapis.com/tf-datasets/titanic/test.csv'

import tensorflow as tf
from tensorflow import keras

TRAIN_URL = 'http://storage.googleapis.com/tf-datasets/titanic/train.csv'
TEST_URL =  'http://storage.googleapis.com/tf-datasets/titanic/eval.csv'

train = keras.utils.get_file("train.csv", TRAIN_URL)
test = keras.utils.get_file("test.csv", TEST_URL)

# print the paths where these files are stored locally
print(train)
print(test)

# does not load the full file in memory, 
# but rather reads in batches when .take(n) is called
# make_csv_dataset has other interesting parameters, such as to recover from errors, 
# or fill the missing data with a default value
train = tf.data.experimental.make_csv_dataset(train, 10, label_name="survived")

def inspect_batch(dataset):
    
    # read the first batch
    for batch, label in dataset.take(1):
        
        # in batch is everything else, as a dictionary of columns
        for k, v in batch.items():
            print(f'{k} : {v.numpy()}')
            
        # it was imported with label (make_csv_dataset invocation)
        print(f'survived : {label.numpy()}')
        
inspect_batch(train)
```
