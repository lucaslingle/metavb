import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import collections

class VAEModel:
    def __init__(self, hps, name=None):
        self.img_height = hps.img_height
        self.img_width = hps.img_width
        self.img_channels = hps.img_channels
        self.discrete_outputs = hps.discrete_outputs

        self.batch_size = hps.batch_size
        self.episode_len = hps.episode_len

        self.enc_blocks = hps.enc_blocks
        self.dec_blocks = hps.dec_blocks
        self.num_filters = hps.num_filters

        self.code_size = hps.code_size

        self.use_bn = hps.use_bn
        self.lr = hps.lr

        self.global_step = tf.train.get_or_create_global_step()
        self._name = 'VAE' if name is None else name

        with tf.variable_scope(self._name):
            self.input_xs = tf.placeholder(dtype=tf.float32, shape=[None, self.episode_len, self.img_height, self.img_width, self.img_channels])
            self.is_train = tf.placeholder(dtype=tf.bool, shape=[])

            # encode
            batch_size = tf.shape(self.input_xs)[0]
            self.input_xs_batched = tf.reshape(self.input_xs, [-1, self.img_height, self.img_width, self.img_channels])
            self.qz_batched = self.qz(self.input_xs_batched, training=self.is_train)
            self.qZ = tfp.distributions.BatchReshape(self.qz_batched, batch_shape=[-1, self.episode_len])
            self.Mu_Z = self.qZ.mean()
            self.Sigma_Z_diag = self.qZ.variance()

            ## same code template as for memory version. here, it collects z samples and kl divs. 
            z_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            z_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)

            read_loop_init_vars = (0, z_array, z_kl_divs_array)
            read_loop_cond = lambda t, a1, a2: t < self.episode_len
            def read_loop_body(t, z_array, z_kl_divs_array):
                Mu_z = self.Mu_Z[:, t, :]
                Sigma_z_diag = self.Sigma_Z_diag[:, t, :]
                qz = tfp.distributions.MultivariateNormalDiag(loc=Mu_z, scale_diag=tf.sqrt(Sigma_z_diag))
                qz = tfp.distributions.Independent(qz)

                z = qz.sample()

                pz = self.pz(batch_size)

                dkl_z = qz.kl_divergence(pz)

                return (t + 1, z_array.write(t, z), z_kl_divs_array.write(t, dkl_z))

            _, z_samples, z_kl_divs = tf.while_loop(read_loop_cond, read_loop_body, read_loop_init_vars)

            self.z_samples = tf.transpose(z_samples.stack(), [1, 0, 2])  # [B, T, C]
            self.z_kl_divs = tf.transpose(z_kl_divs.stack())  # [B, T]

            # decode
            self.z_samples_batched = tf.reshape(self.z_samples, [-1, self.code_size])  # [B*T, C]
            self.px_given_z_batched = self.px_given_z(self.z_samples_batched, training=self.is_train)  # batch_shape [B*T], element_shape [H, W, C]
            self.px_given_z_ = tfp.distributions.BatchReshape(self.px_given_z_batched, batch_shape=[-1, self.episode_len])  # batch_shape [B, T], element_shape [H, W, C]

            self.log_probs_x_given_z = self.px_given_z_.log_prob(self.input_xs)

            # main objective
            self.elbo_episode = tf.reduce_sum(
                (self.log_probs_x_given_z - self.z_kl_divs), axis=1)  # [B]

            self.elbo = tf.reduce_mean(self.elbo_episode, axis=0)  # []
            self.elbo_per_frame = self.elbo / tf.cast(self.episode_len, dtype=tf.float32)

            self.loss = -self.elbo_per_frame

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            tvars = [v for v in tf.trainable_variables() if v.name.startswith(self._name)]
            self.gradients, _ = zip(*self.optimizer.compute_gradients(self.loss, tvars))

            # use control dependencies on update ops - this is required by batchnorm.
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = self.optimizer.apply_gradients(
                    grads_and_vars=zip(self.gradients, tvars),
                    global_step=self.global_step)

            ## tensorboard summaries
            self.dkl_z_per_frame = tf.reduce_mean(self.z_kl_divs, axis=[0, 1])
            tf.summary.scalar('elbo_per_frame', self.elbo_per_frame)
            tf.summary.scalar('dkl_z_per_frame', self.dkl_z_per_frame)
            self.merged_summaries = tf.summary.merge_all()

            ## misc ops - not used during training.
            self.query_z = tf.placeholder(tf.float32, shape=[None, self.code_size])
            self.query_x = tf.placeholder(tf.float32, shape=[None, self.img_height, self.img_width, self.img_channels])

            # decode - using a given z, compute and sample p(x|z)
            def decode(z):
                px = self.px_given_z(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x

            # decode - using a given x, compute and sample q(z)
            def encode(query_x):
                qz = self.qz(query_x, training=False)
                z = qz.sample()
                return z

            # copy -  using a given x, compute and sample q(z|x), p(x|z)
            def copy(query_x):
                z = encode(query_x)
                x = decode(z)
                return x

            self.decoded_x = decode(self.query_z)
            self.copied_x = copy(self.query_x)

    def qz(self, input_x, training=True):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            input_x = input_x if self.discrete_outputs else (2.0 * input_x - 1.0)
            blocks = [input_x]

            for i in range(0, self.enc_blocks):
                block = self.encoder_block(blocks[i], training=training, name='enc_block_' + str(i + 1))
                blocks.append(block)

            res_tower_output = blocks[self.enc_blocks]
            res_tower_output_flattened = tf.layers.flatten(res_tower_output)

            encoded_x = tf.layers.dense(
                res_tower_output_flattened, units=(2 * self.code_size), use_bias=False, activation=None)

            fc1 = tf.layers.dense(encoded_x, units=(2 * self.code_size), use_bias=True, activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1, units=(2 * self.code_size), use_bias=False, activation=None)

            mu, logsigma = tf.split(fc2, 2, axis=1)
            z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            z_dist = tfp.distributions.Independent(z_dist)
            return z_dist

    def px_given_z(self, z, training=True):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            s = (2 ** self.dec_blocks)
            dec_init_h = int(np.ceil(self.img_height / float(s)))
            dec_init_w = int(np.ceil(self.img_width / float(s)))
            fc_units = int(dec_init_h * dec_init_w * self.num_filters)

            fc = tf.layers.dense(z, units=fc_units, use_bias=False, activation=None)
            fc_3d = tf.reshape(fc, shape=[-1, dec_init_h, dec_init_w, self.num_filters])

            blocks = [fc_3d]

            for i in range(0, self.dec_blocks):
                block = self.decoder_block(blocks[i], training=training, name='dec_block_' + str(i + 1))
                blocks.append(block)

            tower_output = blocks[self.dec_blocks]
            cropped_tower_output = self.maybe_center_crop(tower_output, self.img_height, self.img_width)

            if self.discrete_outputs:
                decoded_logits_x = tf.layers.conv2d(
                    cropped_tower_output, filters=self.img_channels,
                    kernel_size=1, strides=1, padding='same', activation=None)

                x_dist = tfp.distributions.Bernoulli(logits=decoded_logits_x)
                x_dist = tfp.distributions.Independent(x_dist)
                return x_dist
            else:
                decoded_mu_x = tf.layers.conv2d(
                    cropped_tower_output, filters=self.img_channels,
                    kernel_size=1, strides=1, padding='same', activation=tf.nn.sigmoid)

                decoded_sigma_x = 5e-3 + 0.5 * tf.layers.conv2d(
                    cropped_tower_output, filters=self.img_channels,
                    kernel_size=1, strides=1, padding='same', activation=tf.nn.sigmoid)

                x_dist = tfp.distributions.MultivariateNormalDiag(loc=decoded_mu_x, scale_diag=decoded_sigma_x)
                x_dist = tfp.distributions.Independent(x_dist)
                return x_dist

    def maybe_center_crop(self, tensor, target_height, target_width):
        with tf.variable_scope('maybe_center_crop', reuse=tf.AUTO_REUSE):
            # assumes tensor shape is such that the extra space is divisible by two
            shape = tensor.get_shape().as_list()
            print(shape)
            h, w = shape[1], shape[2]
            h_slack = h - target_height
            w_slack = w - target_width
            h_start_idx = 0 + tf.cast(h_slack / 2, dtype=tf.int32)
            h_end_idx = h - tf.cast(h_slack / 2, dtype=tf.int32)
            w_start_idx = 0 + tf.cast(w_slack / 2, dtype=tf.int32)
            w_end_idx = w - tf.cast(w_slack / 2, dtype=tf.int32)
            tensor_maybe_cropped = tensor[:, h_start_idx:h_end_idx, w_start_idx:w_end_idx, :]
            return tensor_maybe_cropped

    def maybe_batchnorm(self, inputs, training, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if self.use_bn:
                bn = tf.layers.batch_normalization(inputs, training=training)
            else:
                bn = tf.identity(inputs)
            return bn

    def encoder_block(self, inputs, training, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            glorot_init = tf.glorot_normal_initializer()
            he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
                mode='FAN_IN', uniform=False)

            conv0 = tf.layers.conv2d(
                inputs, filters=self.num_filters, kernel_size=4, strides=2, padding='valid',
                kernel_initializer=glorot_init, activation=None)

            ## residual block without bottleneck
            conv1 = tf.layers.conv2d(
                conv0, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            bn1 = self.maybe_batchnorm(conv1, training=training, name='bn1')
            act1 = tf.nn.relu(bn1)

            conv2 = tf.layers.conv2d(
                act1, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            bn2 = self.maybe_batchnorm(conv2, training=training, name='bn2')

            h = tf.nn.relu(conv0 + bn2)
            return h

    def decoder_block(self, inputs, training, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            glorot_init = tf.glorot_normal_initializer()
            he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
                mode='FAN_IN', uniform=False)

            dconv0 = tf.layers.conv2d_transpose(
                inputs, filters=self.num_filters, kernel_size=4, strides=2, padding='same',
                kernel_initializer=glorot_init, activation=None)

            ## residual block without bottleneck, except using transpose convolutions
            dconv1 = tf.layers.conv2d_transpose(
                dconv0, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            bn1 = self.maybe_batchnorm(dconv1, training=training, name='bn1')
            act1 = tf.nn.relu(bn1)

            dconv2 = tf.layers.conv2d_transpose(
                act1, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            bn2 = self.maybe_batchnorm(dconv2, training=training, name='bn2')

            h = tf.nn.relu(dconv0 + bn2)
            return h

    def get_pz_logsigma(self):
        with tf.variable_scope('pz_logsigma', reuse=tf.AUTO_REUSE):
            init = tf.zeros_initializer()
            pz_logsigma = tf.get_variable(
                name='pz_logsigma', dtype=tf.float32, shape=[], initializer=init, trainable=False)
            return pz_logsigma

    def pz(self, batch_size):
        with tf.variable_scope('pz', reuse=tf.AUTO_REUSE):
            mu = tf.zeros(dtype=tf.float32, shape=[batch_size, self.code_size])
            logsigma = self.get_pz_logsigma() * tf.ones_like(mu)
            z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            z_dist = tfp.distributions.Independent(z_dist)
            return z_dist

    def train(self, sess, input_xs):
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: True
        }
        _ = sess.run(self.train_op, feed_dict=feed_dict)

        elbo_per_frame, step, summaries = sess.run(
            [self.elbo_per_frame, self.global_step, self.merged_summaries], 
            feed_dict=feed_dict)
        
        return elbo_per_frame, step, summaries

    def evaluate(self, sess, input_xs):
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: False
        }
        elbo_per_frame = sess.run(self.elbo_per_frame, feed_dict=feed_dict)
        return elbo_per_frame

    def generate(self, sess, num_samples):
        z_samples = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, self.code_size))
        feed_dict = {
            self.query_z: z_samples,
            self.is_train: False
        }
        generated_x = sess.run(self.decoded_x, feed_dict=feed_dict)
        return generated_x

    def copy(self, sess, x):
        feed_dict = {
            self.query_x: x,
            self.is_train: False
        }
        copied_x = sess.run(self.copied_x, feed_dict=feed_dict)
        return copied_x
