import tensorflow as tf
import numpy as np
from common.deep_learning.basic_model import BasicModel
import common.deep_learning.ops as ops


class KSpaceSuperResolutionMC(BasicModel):
    """
    Represents k-space super resolution model
    """
    def __init__(self, input=None, labels=None, dims_in=None, dims_out=None, batch_size=None,
                 reg_w=0.0, train_phase=None):
        """
        :param input:
        :param labels:
        :param dims_in:
        :param dims_out:
        """
        BasicModel.__init__(self, input=input, labels=labels, dims_in=dims_in, dims_out=dims_out)
        self.train_phase = train_phase
        self.predict = None
        self.loss = None
        self.train_step = None
        self.evaluation = None
        self.x_input_upscale = None
        self.debug = tf.shape(input)
        self.batch_size = batch_size
        self.reg_w = reg_w
        self.regularization_values = []
        self.regularization_sum = None

    def build(self, FLAGS):
        """None
        Build the model graph
        :return:
        """
        with tf.name_scope('model'):
            self.predict = self.__model__()
            if len(self.regularization_values) > 0:
                self.regularization_sum = sum(self.regularization_values)

        with tf.name_scope('loss'):
            self.loss = self.__loss__(predict=self.predict, labels=self.labels, reg=self.regularization_sum)

        with tf.name_scope('train'):
            # Training evaluation
            # Using Adam solver with cross entropy minimize
            self.train_step = self.__training__(s_loss=self.loss, learning_rate=FLAGS.learning_rate)

        with tf.name_scope('evaluation'):
            # Calculate accuracy
            self.evaluation = self.__evaluation__(predict=self.predict, labels=self.labels)

    def __model__(self):
        """
        Define the model
        """
        # Apply image resize for debugging
        self.x_input_upscale = tf.image.resize_bilinear(self.input, np.array([self.dims_out[0],
                                          self.dims_out[1]]), align_corners=None, name='x_input_upscale')

        # Dump input image out
        input_real = tf.slice(self.input, begin=[0,0,0,0], size=[1,-1,-1,1], name='Slice_real_input')
        input_imag = tf.slice(self.input, begin=[0,0,0,1], size=[1,-1,-1,1], name='Slice_imag_input')
        tf.image_summary('x_input_real', input_real)
        tf.image_summary('x_input_imag', input_imag)

        # Model convolutions
        out_dim = 8
        self.conv_1, reg_1 = ops.conv2d(self.input, output_dim=out_dim, k_h=5, k_w=5, d_h=1, d_w=1, name="conv_1")
        # self.debug = tf.contrib.layers.python.layers.utils.constant_value(self.input)
        self.conv_1_bn = ops.batch_norm(self.conv_1, self.train_phase, "bn1")
        self.relu_1 = tf.nn.relu(self.conv_1_bn)
        self.regularization_values.append(reg_1)

        # deconv for get bigger image
        out_shape = [self.batch_size, self.dims_out[0], self.dims_out[1], 4]
        self.conv_2, reg_2 = ops.conv2d_transpose(self.relu_1, output_shape=out_shape,
                                           k_h=3, k_w=3, d_h=2, d_w=1, name="conv_2")
        self.conv_2_bn = ops.batch_norm(self.conv_2, self.train_phase, "bn2")
        # self.conv_2 = ops.conv2d(self.relu_1, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_2")
        self.relu_2 = tf.nn.relu(self.conv_2_bn)
        self.regularization_values.append(reg_2)

        out_dim = 2
        self.conv_3, reg_3 = ops.conv2d(self.relu_2, output_dim=out_dim, k_h=1, k_w=1, d_h=1, d_w=1, name="conv_3")
        self.conv_3_bn = ops.batch_norm(self.conv_3, self.train_phase, "bn3")
        self.relu_3 = tf.nn.relu(self.conv_3_bn)
        self.regularization_values.append(reg_3)

        out_dim = 2
        self.conv_4, reg_4 = ops.conv2d(self.relu_3, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_4")
        self.regularization_values.append(reg_4)

        predict = tf.reshape(self.conv_4, [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='predict')
        # Dump prediction out
        predict_real = tf.slice(predict, begin=[0,0,0,0], size=[1,-1,-1,1], name='Slice_real_input')
        predict_imag = tf.slice(predict, begin=[0,0,0,1], size=[1,-1,-1,1], name='Slice_imag_input')
        tf.image_summary('predict_real', predict_real)
        tf.image_summary('predict_imag', predict_imag)

        return predict  # Sum the reg term in the loss

    def __loss__(self, predict, labels, reg=None):
        """
        Return loss value
        :param predict: prediction from the model
        :param labels: labels input
        :param reg: regularization term
        :return:
        """
        s_loss = tf.reduce_mean(tf.square(tf.squeeze(predict) - labels), name='Loss')
        _ = tf.scalar_summary('square-loss', s_loss)

        if reg is not None:
            tf.scalar_summary('regularization', reg)
            # Add the regularization term to the loss.
            s_loss += self.reg_w * reg
            tf.scalar_summary('square-loss + regularization', s_loss)

        return s_loss

    def __training__(self, s_loss, learning_rate):
        """
        :param s_loss:
        :param learning_rate:
        :return:
        """
        # Add a scalar summary for the snapshot loss.
        # tf.scalar_summary(s_loss.op.name, s_loss)
        # Create Adam optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(s_loss, global_step=global_step)
        return train_op

    def __evaluation__(self, predict, labels):
        """
        :param predict:
        :param labels:
        :return:
        """

        evalu = tf.reduce_mean(tf.square(tf.squeeze(predict) - tf.squeeze(labels)))
        # tf.scalar_summary("Evaluation", evalu)
        # Return the number of true entries.
        return evalu
