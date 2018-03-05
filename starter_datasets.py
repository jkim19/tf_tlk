"""
starter_datasets.py

common ML datasets
    MNIST
    FASHION-MNIST

    Iris
    CIFAR10

Will Jinwoo Kim (Numberseed)
twitter: @numberseedsoft
github: jkim19
"""

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

import iris_data

# mnist dataset
HOMEPAGE = "http://yann.lecun.com/exdb/mnist/"
MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"

# fashion-mnist dataset
HOMEPAGE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
FASHION_MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
FASHION_MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
FASHION_MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"

batch_size = 128
# following with download data if not present
# mnist.train, mnist.test, mnist.validation are DataSets
# each DataSet has images and lables which are np.arrays
# validation dataset is hardcoded as 5000
# train.images.shape (55000, 784)
# train.labels.shape (55000, 10)
# test.images.shape (10000, 784)
# test.labels.shape (10000, 10)
# validation.images.shape (5000, 784)
# validation.images.shape (5000, 10)
mnist = input_data.read_data_sets("data", one_hot=True)
batch_x, batch_y = mnist.train.next_batch(batch_size)

# train, test are tuples
# train[0].shape (60000, 28, 28), train[1].shape (60000,)
# test[0].shape (10000, 28, 28), test[1].shape (10000,)
train, test = tf.keras.datasets.mnist.load_data()
mnist_x, mnist_y = train
mnist_ds = tf.data.Dataset.from_tensor_slices(mnist_x)
print(mnist_ds)

st = time.time()
# downloads to a directory MNIST-data
#   afterwards mnist is the same as input_data's method
mnist = tf.contrib.learn.datasets.load_dataset('mnist')
train_data = mnist.train.images  # returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

(train_x, train_y), (test_x, test_y) = iris_data.load_data()


