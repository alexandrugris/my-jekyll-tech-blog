---
layout: post
title:  "Introduction to TensorFlow"
date:   2021-03-22 09:15:16 +0200
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