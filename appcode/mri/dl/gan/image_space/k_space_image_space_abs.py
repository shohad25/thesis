import tensorflow as tf
import numpy as np
from common.deep_learning.basic_model import BasicModel
import common.deep_learning.ops as ops


class KSpaceSuperResolutionWGAN(BasicModel):
    """
    Represents k-space super resolution model
    """

    def __init__(self, input=None, labels=None, dims_in=None, dims_out=None, FLAGS=None, train_phase=None, adv_loss_w=None):
        """
        :param input:
        :param labels:
        :param dims_in:
        :param dims_out:
        """
        BasicModel.__init__(self, input=labels, labels=labels, dims_in=dims_in, dims_out=dims_out)
        self.input = labels
        print("Input is y label")
        # HACK
        self.input_d = labels
        # HACK

        self.FLAGS = FLAGS
        self.train_phase = train_phase
        self.predict_g = None
        self.adv_loss_w = adv_loss_w

        self.train_op_g = None
        self.train_op_d = None

        self.evaluation = None
        self.update_ops = None
        self.x_input_upscale = {}
        self.debug = None
        self.batch_size = self.FLAGS.mini_batch_size
        # self.reg_w = self.FLAGS.regularization_weight
        self.regularization_values = []
        self.regularization_sum = None

        self.regularization_values_d = []
        self.regularization_sum_d = None

        self.clip_weights = None

        tf.get_collection('G')

    def build(self):
        """None
        Build the model graph
        :return:
        """
        with tf.name_scope('G_'):
            self.predict_g = self.__G__()

        with tf.name_scope('loss'):
            # self.loss_g = self.__loss_g__(predict=self.predict_g, self.labels, reg=self.regularization_sum)
            self.__loss__()

        with tf.name_scope('training'):
            self.train_op_g = self.__training__(learning_rate=self.FLAGS.learning_rate)

        with tf.name_scope('evaluation'):
            # Calculate accuracy L2 norm
            self.evaluation = self.__evaluation__(predict=self.predict_g, labels=self.labels)

    def __G__(self):
        """
        Define the model
        """
        x_real = self.input['real'] * self.input['mask']
        x_imag = self.input['imag'] * self.input['mask']

        # if self.FLAGS.gen_loss_adversarial > 0:
        #     mask_not = tf.cast(tf.logical_not(tf.cast(self.labels['mask'], tf.bool)), tf.float32)
        #     print "Noise level: (-0.01,0.01)"
        #     minval = -0.01
        #     maxval = 0.01
        #     noise_real = mask_not * tf.random_uniform(shape=tf.shape(x_real), minval=minval, maxval=maxval,
        #                                                 dtype=tf.float32, seed=None, name='z_real')
        #     noise_imag = mask_not * tf.random_uniform(shape=tf.shape(x_real), minval=minval, maxval=maxval,
        #                                                 dtype=tf.float32, seed=None, name='z_imag')
        #     x_real += noise_real
        #     x_imag += noise_imag

        # input image is zero padded image
        input_image = self.get_reconstructed_image(real=x_real, imag=x_imag, name='Both')
        input_image = tf.abs(tf.complex(real=tf.expand_dims(input_image[:,0,:,:], axis=1), imag=tf.expand_dims(input_image[:,1,:,:], axis=1)))

        # Create the inputs
        ref_image = self.get_reconstructed_image(real=self.input['real'], imag=self.input['imag'], name='Both')
        ref_image = tf.abs(tf.complex(real=tf.expand_dims(ref_image[:,0,:,:], axis=1), imag=tf.expand_dims(ref_image[:,1,:,:], axis=1)))
        self.ref_image = ref_image
        tf.summary.image('G_reference', tf.transpose(ref_image, (0, 2, 3, 1)), collections='G', max_outputs=4)

        if self.FLAGS.dump_debug:
            zero_image = tf.abs(tf.complex(real=tf.expand_dims(input_image[:, 0, :, :], axis=1),
                                          imag=tf.expand_dims(input_image[:, 1, :, :], axis=1)))
            tf.summary.image('G_zero', tf.transpose(zero_image, (0, 2, 3, 1)), collections='G', max_outputs=4)

        self.x_input_upscale['real'] = x_real
        self.x_input_upscale['imag'] = x_imag

        # Model convolutions
        input_image_reshaped = tf.reshape(tf.transpose(input_image, (0,2,3,1)), (-1,256,256,1))

        input_split = tf.extract_image_patches(images=input_image_reshaped, ksizes=[1, 2, 2, 1],
                                               strides=[1, 1, 1, 1], rates=[1, 128, 128, 1], padding='VALID')

        input_split = tf.transpose(tf.reshape(input_split, (-1,128,128,1)), (0,3,1,2))
        # for i in range(0,4):
        #     tf.summary.image('G_test', tf.expand_dims(test[:,:,:,i], axis=3), collections='G', max_outputs=1)

        # tf.summary.image('G_merged', merged, collections='G', max_outputs=1)

        out_dim = 16
        self.conv_1 = ops.conv2d(input_split, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_1")
        self.conv_1_bn = ops.batch_norm(self.conv_1, self.train_phase, decay=0.98, name="G_bn1")
        self.relu_1 = tf.nn.relu(self.conv_1_bn)

        out_dim = 32
        self.conv_2 = ops.conv2d(self.relu_1, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_2")
        self.conv_2_bn = ops.batch_norm(self.conv_2, self.train_phase, decay=0.98, name="G_bn2")
        self.relu_2 = tf.nn.relu(self.conv_2_bn)

        out_dim = 64
        self.conv_3 = ops.conv2d(self.relu_2, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_3")
        self.conv_3_bn = ops.batch_norm(self.conv_3, self.train_phase, decay=0.98, name="G_bn3")
        self.relu_3 = tf.nn.relu(self.conv_3_bn)

        out_dim = 32
        self.conv_4 = ops.conv2d(self.relu_3, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_4")
        self.conv_4_bn = ops.batch_norm(self.conv_4, self.train_phase, decay=0.98, name="G_bn4")
        self.relu_4 = tf.nn.relu(self.conv_4_bn)

        out_dim = 8
        self.conv_5 = ops.conv2d(self.relu_4, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_5")
        self.conv_5_bn = ops.batch_norm(self.conv_5, self.train_phase, decay=0.98, name="G_bn5")
        self.relu_5 = tf.nn.relu(self.conv_5_bn)

        out_dim = 1
        self.conv_6 = ops.conv2d(self.relu_5, output_dim=out_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="G_conv_6")

        predict = {}

        out_reshaped = tf.reshape(self.conv_6, (-1, 4, 128, 128))
        upper = tf.concat([tf.expand_dims(out_reshaped[:,0,:,:], axis=1), tf.expand_dims(out_reshaped[:,1,:,:], axis=1)], axis=3)
        lower = tf.concat([tf.expand_dims(out_reshaped[:,2,:,:], axis=1), tf.expand_dims(out_reshaped[:,3,:,:], axis=1)], axis=3)
        merged = tf.concat([upper, lower], axis=2)

        predict['real'] = merged

        # with tf.name_scope("final_predict"):
        #     predict['real'] = tf.add(predict['real'], input_image, name='real')

        tf.add_to_collection("predict", predict['real'])

        # Dump prediction out
        out_image = predict['real']
        tf.summary.image('G_predict', tf.transpose(out_image, (0, 2, 3, 1)), collections='G', max_outputs=4)

        return predict

    def __loss__(self):
        """
        Calculate loss
        :return:
        """
        # Context loss L2
        print("You are using L2 loss")
        self.context_loss = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(self.predict_g['real'] - self.ref_image)))

        self.g_loss = self.FLAGS.gen_loss_context * self.context_loss
        tf.summary.scalar('g_loss', self.g_loss, collections='G')

        self.reg_loss_g = self.get_weights_regularization(dump=self.FLAGS.dump_debug, collection='G')
        self.g_loss_no_reg = self.g_loss
        self.g_loss += self.reg_loss_g
        if self.FLAGS.dump_debug:
            tf.summary.scalar('g_loss_plus_reg', self.g_loss, collections='G')
            tf.summary.scalar('g_loss_reg_only', self.reg_loss_g, collections='D')

    def __training__(self, learning_rate):
        """
        :param learning_rate:
        :return:

        """
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'G_' in var.name]

        # Create RMSProb optimizer with the given learning rate.
        optimizer_g = tf.train.RMSPropOptimizer(self.FLAGS.learning_rate, centered=True)

        # Create a variable to track the global step.
        global_step_g = tf.Variable(0, name='global_step_g', trainable=False)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        grad_g = optimizer_g.compute_gradients(loss=self.g_loss, var_list=self.g_vars)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Ensures that we execute the update_ops before performing the train_step
        with tf.control_dependencies(self.update_ops):
            train_op_g = optimizer_g.apply_gradients(grad_g, global_step=global_step_g)

        return train_op_g

    def __evaluation__(self, predict, labels):
        """
        :param predict:
        :param labels:
        :return:
        """
        evalu = self.context_loss
        return evalu

    def get_reconstructed_image(self, real, imag, name=None):
        """
        :param real:
        :param imag:
        :param name:
        :return:
        """
        complex_k_space_label = tf.complex(real=tf.squeeze(real), imag=tf.squeeze(imag), name=name+"_complex_k_space")
        rec_image_complex = tf.expand_dims(tf.ifft2d(complex_k_space_label), axis=1)
        
        rec_image_real = tf.reshape(tf.real(rec_image_complex), shape=[-1, 1, self.dims_out[1], self.dims_out[2]])
        rec_image_imag = tf.reshape(tf.imag(rec_image_complex), shape=[-1, 1, self.dims_out[1], self.dims_out[2]])

        # Shifting
        top, bottom = tf.split(rec_image_real, num_or_size_splits=2, axis=2)
        top_left, top_right = tf.split(top, num_or_size_splits=2, axis=3)
        bottom_left, bottom_right = tf.split(bottom, num_or_size_splits=2, axis=3)

        top_shift = tf.concat(axis=3, values=[bottom_right, bottom_left])
        bottom_shift = tf.concat(axis=3, values=[top_right, top_left])
        shifted_image = tf.concat(axis=2, values=[top_shift, bottom_shift])

        # Shifting
        top_imag, bottom_imag = tf.split(rec_image_imag, num_or_size_splits=2, axis=2)
        top_left_imag, top_right_imag = tf.split(top_imag, num_or_size_splits=2, axis=3)
        bottom_left_imag, bottom_right_imag = tf.split(bottom_imag, num_or_size_splits=2, axis=3)

        top_shift_imag = tf.concat(axis=3, values=[bottom_right_imag, bottom_left_imag])
        bottom_shift_imag = tf.concat(axis=3, values=[top_right_imag, top_left_imag])
        shifted_image_imag = tf.concat(axis=2, values=[top_shift_imag, bottom_shift_imag])

        shifted_image_two_channels = tf.stack([shifted_image[:,0,:,:], shifted_image_imag[:,0,:,:]], axis=1)
        return shifted_image_two_channels

    def get_weights_regularization(self, dump=False, collection=None):
        """
        Calculate sum of regularization (L2)
        :param dump: dump to tensorboard
        :return:
        """
        if collection is None:
            w_collection = tf.get_collection('regularization_w')
            b_collection = tf.get_collection('regularization_b')
        else:
            w_collection = [var for var in tf.get_collection('regularization_w') if collection in var.name]
            b_collection = [var for var in tf.get_collection('regularization_b') if collection in var.name]

        reg_w = tf.add_n(w_collection, name='regularization_w') if len(w_collection) > 0 else 0
        reg_b = tf.add_n(b_collection, name='regularization_b') if len(b_collection) > 0 else 0

        if dump:
            tf.summary.scalar('Regularization - W', reg_w, collections=collection)
            tf.summary.scalar('Regularization - b', reg_b, collections=collection)

        return self.FLAGS.reg_w * reg_w + self.FLAGS.reg_b * reg_b
