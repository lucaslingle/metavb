import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import collections

ContextState = collections.namedtuple('ContextState', field_names=['mu', 'sigma'])


class NeuralStatistician:
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

        self.encoder_output_size = 2 * hps.code_size
        self.context_size = hps.context_size
        self.code_size = hps.code_size

        self.use_bn = hps.use_bn
        self.trainable_context = hps.trainable_context
        self.activation = lambda vec: tf.nn.relu(vec)

        self.lr = hps.lr

        self.global_step = tf.train.get_or_create_global_step()
        self._name = 'NS' if name is None else name

        with tf.variable_scope(self._name):
            self.input_xs = tf.placeholder(dtype=tf.float32, shape=[None, self.episode_len, self.img_height, self.img_width, self.img_channels])
            self.is_train = tf.placeholder(dtype=tf.bool, shape=[])

            # encode
            batch_size = tf.shape(self.input_xs)[0]
            self.input_xs_batched = tf.reshape(self.input_xs, [-1, self.img_height, self.img_width, self.img_channels])
            self.encoded_xs_batched = self.encode(self.input_xs_batched, training=self.is_train)
            self.encoded_xs = tf.reshape(self.encoded_xs_batched, [-1, self.episode_len, self.encoder_output_size])

            self.qC = self.qC_given_X(self.encoded_xs)
            self.qC_mu = self.qC.mean()
            self.qC_sigma = self.qC.stddev()

            ## read from context
            # found it helpful to sample context variable c separately for each t in the episode. 
            # note that this does not change the expected value
            z_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            z_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)

            read_loop_init_vars = (0, z_array, z_kl_divs_array)
            read_loop_cond = lambda t, a1, a2: t < self.episode_len
            def read_loop_body(t, z_array, z_kl_divs_array):
                enc_x = self.encoded_xs[:, t, :]
                C = self.qC.sample()

                qz = self.qz_given_xC(enc_x, C)
                z = qz.sample()

                # computing single-sample estimator of E_{q(C|X)}[D_{KL}(q(z|x,C)||p(z|C))].
                pz = self.pz_given_C(C)
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

            # dkl_C
            self.pC = self.get_context_prior(batch_size)
            self.dkl_C = self.qC.kl_divergence(self.pC)

            # objective
            self.elbo_episode = tf.reduce_sum(
                (self.log_probs_x_given_z - self.z_kl_divs), axis=1) - self.dkl_C  # [B]

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
            self.dkl_C_per_frame = tf.reduce_mean(self.dkl_C, axis=0) / tf.cast(self.episode_len, dtype=tf.float32)
            tf.summary.scalar('elbo_per_frame', self.elbo_per_frame)
            tf.summary.scalar('dkl_z_per_frame', self.dkl_z_per_frame)
            tf.summary.scalar('dkl_C_per_frame', self.dkl_C_per_frame)
            self.merged_summaries = tf.summary.merge_all()

            ## misc ops - not used during training.
            #  using a given memory state:
            self.input_c_mu = tf.placeholder(tf.float32, shape=[None, self.context_size])
            self.input_c_sigma = tf.placeholder(tf.float32, shape=[None, self.context_size])
            self.context_state_ = tfp.distributions.MultivariateNormalDiag(loc=self.input_c_mu, scale_diag=self.input_c_sigma)
            self.context_state = tfp.distributions.Independent(self.context_state_)

            # generate - q(C|X), p(z|C), p(x|z)
            def generate(qC):
                C = qC.mean() # qC.sample()
                pz = self.pz_given_C(C)
                z = pz.sample()
                px = self.px_given_z(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x

            self.generated_x = generate(self.context_state)

            # read - q(C|X), q(z|x,C), p(x|z)
            def read(query_x, qC):
                C = qC.mean() # qC.mean()
                enc_x = self.encode(query_x, training=False)
                qz = self.qz_given_xC(enc_x, C)
                z = qz.sample()
                px = self.px_given_z(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x

            self.query_x = tf.placeholder(tf.float32, shape=[None, self.img_height, self.img_width, self.img_channels])
            self.read_x = read(self.query_x, self.context_state)

    def get_context_prior(self, batch_size):
        with tf.variable_scope('initial_state', reuse=tf.AUTO_REUSE):

            mu_params = tf.get_variable(
                name='mu_params', dtype=tf.float32, shape=[self.context_size], 
                initializer=tf.zeros_initializer(), trainable=self.trainable_context)

            logsigma_params = tf.get_variable(
                name='logsigma_params', dtype=tf.float32, shape=[self.context_size], 
                initializer=tf.zeros_initializer(), trainable=self.trainable_context)

            mu = mu_params
            logsigma = logsigma_params

            mu = tf.tile(tf.expand_dims(mu, 0), [batch_size, 1])
            logsigma = tf.tile(tf.expand_dims(logsigma, 0), [batch_size, 1])

            c_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            c_dist = tfp.distributions.Independent(c_dist)
            return c_dist

    def qC_given_X(self, encoded_xs):
        with tf.variable_scope('qC_given_X', reuse=tf.AUTO_REUSE):
            ##pooling_dim = 1000 # Appendix B, Edwards and Storkey
            pooling_dim = self.encoder_output_size # this works

            W1 = tf.get_variable(dtype=tf.float32, shape=[self.encoder_output_size, pooling_dim], 
                name='W1', trainable=True)

            b1 = tf.get_variable(dtype=tf.float32, shape=[pooling_dim], name='b1', trainable=True)
            pooling_space_encoded_xs = tf.einsum('bte,ep->btp', encoded_xs, W1) + tf.reshape(b1, [1, 1, pooling_dim])
            #pooling_space_encoded_xs = tf.nn.elu(pooling_space_encoded_xs) # Appendix B, Edwards and Storkey. gave nan losses
            pooling_space_encoded_xs = self.activation(pooling_space_encoded_xs)

            pooled = tf.reduce_mean(pooling_space_encoded_xs, axis=-1) # average pooling. shape [b, p]

            ##setting that worked: 
            fc1 = tf.layers.dense(pooled, units=(2 * self.context_size), activation=self.activation)
            fc2 = tf.layers.dense(fc1, units=(2 * self.context_size), activation=None)
            mu, logsigma = tf.split(fc2, 2, axis=-1)

            # Appendix B, Edwards and Storkey
            # gave nan losses
            #fc = tf.layers.dense(pooled, units=(2 * self.context_size), activation=None)
            #mu, logsigma = tf.split(fc, 2, axis=-1)

            C_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            C_dist = tfp.distributions.Independent(C_dist)
            return C_dist

    def encode(self, input_x, training=True):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            input_x = input_x if self.discrete_outputs else (2.0 * input_x - 1.0)
            blocks = [input_x]

            for i in range(0, self.enc_blocks):
                block = self.encoder_block(blocks[i], training=training, name='enc_block_' + str(i + 1))
                blocks.append(block)

            tower_output = blocks[self.enc_blocks]
            tower_output_flattened = tf.layers.flatten(tower_output)

            encoded_x = tf.layers.dense(
                tower_output_flattened, units=self.encoder_output_size, use_bias=False, activation=None)

            return encoded_x

    def qz_given_xC(self, enc_x, C):
        with tf.variable_scope('qz_given_xC', reuse=tf.AUTO_REUSE):
            vec = tf.concat([enc_x, C], axis=-1)
            ##hidden_dim = 1000 # Appdx B, Edwards and Storkey. gave nan losses
            hidden_dim = (2 * self.code_size)
            fc1 = tf.layers.dense(vec, units=hidden_dim, use_bias=True, activation=self.activation) #activation=tf.nn.elu) ## Appdx B, Edwards and Storkey. gave nan losses
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

                decoded_sigma_x = 5e-3 + 0.50 * tf.layers.conv2d(
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
            he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)

            conv0 = tf.layers.conv2d(
                inputs, filters=self.num_filters, kernel_size=4, strides=2, padding='valid',
                kernel_initializer=glorot_init, activation=None)

            ## residual block without bottleneck
            conv1 = tf.layers.conv2d(
                conv0, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            bn1 = self.maybe_batchnorm(conv1, training=training, name='bn1')
            act1 = self.activation(bn1)

            conv2 = tf.layers.conv2d(
                act1, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            bn2 = self.maybe_batchnorm(conv2, training=training, name='bn2')

            h = self.activation(conv0 + bn2)
            return h

    def decoder_block(self, inputs, training, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            glorot_init = tf.glorot_normal_initializer()
            he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)

            dconv0 = tf.layers.conv2d_transpose(
                inputs, filters=self.num_filters, kernel_size=4, strides=2, padding='same',
                kernel_initializer=glorot_init, activation=None)

            ## residual block without bottleneck, except using transpose convolutions
            dconv1 = tf.layers.conv2d_transpose(
                dconv0, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            bn1 = self.maybe_batchnorm(dconv1, training=training, name='bn1')
            act1 = self.activation(bn1)

            dconv2 = tf.layers.conv2d_transpose(
                act1, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            bn2 = self.maybe_batchnorm(dconv2, training=training, name='bn2')

            h = self.activation(dconv0 + bn2)
            return h

    def pz_given_C(self, C):
        with tf.variable_scope('pz_given_C', reuse=tf.AUTO_REUSE):
            ##hidden_dim = 1000 # Appdx B, Edwards and Storkey. ## gave nan losses
            hidden_dim = (2 * self.code_size)
            fc1 = tf.layers.dense(C, units=hidden_dim, activation=self.activation) # Appdx B, Edwards and Storkey. ## gave nan losses. #activation=tf.nn.elu)
            fc2 = tf.layers.dense(fc1, units=(2 * self.code_size), activation=None)
            mu, logsigma = tf.split(fc2, 2, axis=-1)
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

    def get_context_state(self, sess, input_xs):
        # this uses input_xs to compute variational posterior q(C|X)
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: False
        }
        mu, sigma = sess.run([self.qC_mu, self.qC_sigma], feed_dict=feed_dict)
        return ContextState(mu=mu, sigma=sigma)

    def generate_from_context_state(self, sess, context_state):
        feed_dict = {
            self.input_c_mu: context_state.mu,
            self.input_c_sigma: context_state.sigma,
            self.is_train: False
        }
        generated_x = sess.run(self.generated_x, feed_dict=feed_dict)
        return generated_x

    def read_from_context_state(self, sess, context_state, x):
        feed_dict = {
            self.input_c_mu: context_state.mu,
            self.input_c_sigma: context_state.sigma,
            self.query_x: x,
            self.is_train: False
        }
        read_x = sess.run(self.read_x, feed_dict=feed_dict)
        return read_x


