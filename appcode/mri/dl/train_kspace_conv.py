""" Train to complete k_space lines with factor 2
k_space real + imaginary are scaled with log function
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from common.deep_learning.helpers import *

# k space data set
base_dir = '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/03_01_2016/shuffle/'
file_names = ['k_space_real', 'k_space_imag', 'mask', 'meta_data', 'image']

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('mini_batch_size', 50, 'Size of mini batch')
flags.DEFINE_integer('print_test', 100, 'Print test frequancy')

def main(_):
    # Import data
    data_set = KspaceDataSet(base_dir, file_names, stack_size=50)
    sess = tf.InteractiveSession()

    # Init inputs as patheholders
    x = tf.placeholder(tf.float32, shape=[None, 256, 256], name='x_input')
    y_ = tf.placeholder(tf.float32, shape=[None, 256, 256], name='y_input')

    # Reshape the input for batchSize, 28x28 image, 1 channel
    x_image = tf.reshape(x, [-1, 256, 256, 1], name='x_input_reshape')
    y_image = tf.reshape(y_, [-1, 256, 256, 1], name='y_input_reshape')

    image_summary = tf.image_summary('input', x_image)

    # Init all W and b:
    # First convolutional layer weights
    W_conv1 = weight_variable([5, 5, 1, 4])
    b_conv1 = bias_variable([4])

    # Second convolutional layer weights
    W_conv2 = weight_variable([5, 5, 4, 1])
    b_conv2 = bias_variable([1])

    # Network
    # First conv & relu
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Second conv & relu
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    y_pred = tf.reshape(h_conv2, [-1, 256, 256, 1], name='y_pred')


    # More name scopes will clean up the graph representation
    with tf.name_scope('xent'):
        # Loss
        abs_loss = tf.reduce_sum(tf.abs(y_image - y_pred))
        _ = tf.scalar_summary('abs-loss', abs_loss)
    
    with tf.name_scope('train'):
        # Training evaluation
        # Using Adam solver with cross entropy minimize
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(abs_loss)

    with tf.name_scope('test'):
        # Choose the max value for prediction
        correct_prediction = tf.abs(y_image - y_pred)
        # Calculate accuracy
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy = tf.reduce_mean(correct_prediction)
        _ = tf.scalar_summary('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('/tmp/k_space_logs', sess.graph_def)
    tf.initialize_all_variables().run()

    # Train the model, and feed in test data and record summaries every 10 steps

    for i in range(FLAGS.max_steps):
        if i % FLAGS.print_test == 0:  # Record summary data and the accuracy    
            next_batch = data_set.test.next_batch(FLAGS.mini_batch_size)
            batch_xs = next_batch['k_space_real']  # TODO:: add log * mask
            batch_ys = next_batch['k_space_real']
            feed = {x: batch_xs, y_: batch_ys}
            result = sess.run([merged, accuracy], feed_dict=feed)
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            next_batch = data_set.train.next_batch(FLAGS.mini_batch_size)
            batch_xs = next_batch['k_space_real']  # TODO:: add log * mask
            batch_ys = next_batch['k_space_real']
            feed = {x: batch_xs, y_: batch_ys}
            sess.run(train_step, feed_dict=feed)

if __name__ == '__main__':
    tf.app.run()
