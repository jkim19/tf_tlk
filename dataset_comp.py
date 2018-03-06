import time

import tensorflow as tf

# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("data", one_hot=True)
mnist = tf.contrib.learn.datasets.load_dataset('mnist')

# Parameters
learning_rate = 0.01

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
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# VS

# logits
logits = tf.matmul(x, W) + b
# loss
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
loss = tf.reduce_mean(entropy)
# optimizer
tf.train.AdamOptimizer(learning_rate).minimize(loss)

# TODO: compare placeholder vs dataset
# TODO: compare cost vs loss methods

"""

# Start training
with tf.Session() as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())



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