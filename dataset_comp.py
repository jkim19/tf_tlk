import os
import numpy as np
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


mnist = input_data.read_data_sets("data", one_hot=True)
# mnist = input_data.read_data_sets("data")
# vs raw ... trY vs mnist.train.labels (one_hot)
path = 'data'
fd = open(os.path.join(path, 'train-images.idx3-ubyte'))
loaded = np.fromfile(file=fd, dtype=np.uint8)
trainX = loaded[16:].reshape((60000, 784)).astype(np.float32)
# trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)
fd = open(os.path.join(path, 'train-labels.idx1-ubyte'))
loaded = np.fromfile(file=fd, dtype=np.uint8)
trainY = loaded[8:].reshape((60000)).astype(np.int32)
trX = trainX[5000:] / 255.
trY = trainY[5000:]
valX = trainX[:5000, ] / 255.
valY = trainY[:5000]
# vs tf.contrib... which is same as above (raw)
mnist2 = tf.contrib.learn.datasets.load_dataset('mnist')
train_labels = np.asarray(mnist2.train.labels, dtype=np.int32)

# Parameters
learning_rate = 0.01
training_epochs = 100

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784])
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10])

# Set model weights.  trainable=True
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
# Softmax
pred = tf.nn.softmax(tf.matmul(x, W) + b)
# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# ####################################################################
# Start training
st = time.time()
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
training_epochs = 100
with tf.Session() as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={x: mnist.train.images,
                                                      y: mnist.train.labels})
        print("Epoch: {} c={}".format(epoch+1, c))

    print("Optimization finished")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: {}", accuracy.eval({x: mnist.test.images[:3000],
                                         y: mnist.test.labels[:3000]}))
et = time.time()
print("{} minutes".format((et-st)/60.0))
# epochs = 25; accuracy=0.699; GradientDescentOptimizer
# epochs = 25; accuracy=0.860; AdamOptimizer
# epochs = 100; acc = 0.8993; AdamOptimizer; 0.6687 minutes
# ####################################################################

st = time.time()
training_epochs = 25
batch_size = 100
display_step = 1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch: {} cost={}".format(epoch+1, avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: {}", accuracy.eval({x: mnist.test.images[:3000],
                                         y: mnist.test.labels[:3000]}))
et = time.time()
print("{} minutes".format((et-st)/60.0))
# epochs = 25; accuracy=0.890; GradientDescentOptimizer with batches
# acc=0.889; 0.228 minutes
# ####################################################################

# dataset playground
dataset = tf.data.Dataset.from_tensor_slices((mnist.train.labels))
dataset = dataset.batch(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    value = sess.run(next_element)
    print("i:{} value:{} shape:{}".format(i, value, value.shape))
# ####################################################################

# slow. not correct. placeholder vs dataset but this is actually using both
st = time.time()
batch_size = 100
training_epochs = 25
display_step = 1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
dataset = tf.data.Dataset.from_tensor_slices((mnist.train.images,
                                              mnist.train.labels))\
                                              .batch(batch_size)
iterator = dataset.make_initializable_iterator()  # make_one_shot_iterator()
# el = iterator.get_next()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        sess.run(iterator.initializer)
        el = iterator.get_next()
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        while True:
            try:
                a, b = sess.run(el)
                _, c = sess.run([optimizer, cost],
                                feed_dict={x: a.reshape(a.shape[0], 784),
                                           y: b.reshape(b.shape[0], 10)})
                avg_cost += c / total_batch
            except tf.errors.OutOfRangeError:
                break
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch: {} cost={}".format(epoch+1, avg_cost))

    print("Optimization Finished!")
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: {}", accuracy.eval({x: mnist.test.images[:3000],
                                         y: mnist.test.labels[:3000]}))
et = time.time()
print("{} minutes".format((et-st)/60.0))
# accuracy 0.890
# ####################################################################

# VS

# logits
# logits = tf.matmul(x, W) + b
# loss
# entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
# loss = tf.reduce_mean(entropy)
# optimizer
# tf.train.AdamOptimizer(learning_rate).minimize(loss)

# TODO: compare placeholder vs dataset
# TODO: compare cost vs loss methods

"""
with tf.session() as sess:
    for i in range(100):  # 100 epochs
        for x, y in data:
            sess.run(optimizer, feed_dict={X: x, Y: y})

dataset = tf.data.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))
iterator = dataset.make_one_shot_iterator()
X, Y = iterator.get_next()

with tf.Session() as sess:
    print(sess.run([X, Y]))
    print(sess.run([X, Y]))
    print(sess.run([X, Y]))

iterator = dataset.make_initializable_iterator()
for i in range(100):
    sess.run(iterator.initializer)
    total_loss = 0
    try:
        while True:
            sess.run([optimizier])
    except tf.erros.OutOfRangeError:
        pass

"""