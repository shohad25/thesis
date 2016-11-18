import math
import tensorflow as tf
import numpy as np
from common.deep_learning.basic_model import BasicModel
import common.deep_learning.ops as ops

class KSpaceSuperResolution(BasicModel):

    def __init__(self, input=None, labels=None, dims_in=None, dims_out=None):
        BasicModel.__init__(input=input, labels=labels, dims_in=dims_in, dims_out=dims_out)

    def model(self):
        """
        :param x_input: input placeholder
        :param dims: array of dimensions [W,H,C]
        :return:
        """
        # Reshape the input for batchSize, dims_in[0] X dims_in[1] image, dims_in[2] channels
        x_image = tf.reshape(BasicModel.input, [-1, BasicModel.dims_in[0],
                             BasicModel.dims_in[1], BasicModel.dims_in[2]],
                             name='x_input_reshaped')

        # Apply image resize
        x_image_upscaled = tf.image.resize_bilinear(x_image, np.array([BasicModel.dims_out[0],
                                                    BasicModel.dims_out[1]]), align_corners=None,
                                                    name='x_input_upscaled')
        # Dump input image out
        tf.image_summary('x_upscaled', x_image_upscaled)

        # Model convolutions
        conv_1 = ops.conv2d(x_image_upscaled, output_dim=8, k_h=5, k_w=5, name="conv_1")
        relu_1 = tf.nn.relu(conv_1)

        conv_2 = ops.conv2d(relu_1, output_dim=4, k_h=3, k_w=3, name="conv_2")
        relu_2 = tf.nn.relu(conv_2)

        conv_3 = ops.conv2d(relu_2, output_dim=1, k_h=1, k_w=1, name="conv_3")
        relu_3 = tf.nn.relu(conv_3)

        conv_4 = ops.conv2d(relu_3, output_dim=1, k_h=3, k_w=3, name="conv_4")

        predict = tf.reshape(conv_4, [-1, 256, 256, 1], name='predict')

        # Dump prediction out
        tf.image_summary('predict', predict)

        return predict


def loss(predict, labels):
    """
    Return loss value
    :param predict: prediction from the model
    :param labels: labels input
    :return:
    """
    loss = tf.reduce_mean(tf.square(predict - labels), name='Loss')
    _ = tf.scalar_summary('square-loss', loss)
    return loss


def training(loss, learning_rate):
    """
    :param loss:
    :param learning_rate:
    :return:
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create Adam optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(predict, labels):
    """

    :param predict:
    :param labels:
    :return:
    """
    evalu = tf.reduce_mean(tf.square(predict - labels))
    tf.scalar_summary("Evaluation", evalu)
    # Return the number of true entries.
    return evalu
