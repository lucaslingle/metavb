import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import collections

MemoryState = collections.namedtuple('MemoryState', field_names=['R', 'U'])

class DynamicKanervaMachine:
    def __init__(self, hps, name=None):
        self.img_height = hps.img_height
        self.img_width = hps.img_width
        self.img_channels = hps.img_channels
        self.discrete_outputs = hps.discrete_outputs

        self.episode_len = hps.episode_len

        self.encoder_output_dim = hps.code_size

        self.memory_size = hps.memory_size
        self.code_size = hps.code_size

        self.enc_blocks = hps.enc_blocks
        self.dec_blocks = hps.dec_blocks
        self.num_filters = hps.num_filters

        self.sample_memory = hps.sample_memory
        self.use_bn = hps.use_bn
        self.trainable_memory = hps.trainable_memory

        self.lr = hps.lr

        self.global_step = tf.train.get_or_create_global_step()
        self._name = 'DynamicKanervaMachine' if name is None else name

        with tf.variable_scope(self._name):
            self.input_xs = tf.placeholder(dtype=tf.float32, shape=[None, self.episode_len, self.img_height, self.img_width, self.img_channels])
            self.is_train = tf.placeholder(dtype=tf.bool, shape=[]) # if false, batch norm uses accumulated stats.

            # encode
            self.input_xs_batched = tf.reshape(self.input_xs, [-1, self.img_height, self.img_width, self.img_channels])
            self.encoded_xs_batched = self.encode(self.input_xs_batched, training=self.is_train)
            self.encoded_xs = tf.reshape(self.encoded_xs_batched, [-1, self.episode_len, self.encoder_output_dim]) # [B, T, E]
            batch_size = tf.shape(self.encoded_xs)[0]

            # writing algorithm for the DKM - computes a distribution q(M) using enc_x and deterministic point estimate Mu_w for q(w_t)
            self.initial_state = self.get_initial_state(batch_size)
            write_loop_init_vars = (0, self.initial_state)
            write_loop_cond = lambda t, m: t < self.episode_len
            def write_loop_body(t, memory_state):
                enc_x = self.encoded_xs[:, t, :]
                qw = self.qw(enc_x, memory_state.R)
                w = qw.mean()
                z = enc_x

                R = memory_state.R
                U = memory_state.U
                Delta = z - tf.einsum('bk,bkc->bc', w, R)
                Sigma_c = tf.einsum('bk,bkl->bl', w, U)
                wUwT = tf.einsum('bk,bk->b', Sigma_c, w)
                Sigma_xi = tf.exp(2.0 * self.get_pz_logsigma())
                Sigma_z = wUwT + Sigma_xi
                Sigma_z_inv = 1.0 / Sigma_z
                Sigma_c_transpose_Sigma_z_inv = tf.einsum('bk,b->bk', Sigma_c, Sigma_z_inv)
                R_new = R + tf.einsum('bk,bc->bkc', Sigma_c_transpose_Sigma_z_inv, Delta)
                U_new = U - tf.einsum('bk,bl->bkl', Sigma_c_transpose_Sigma_z_inv, Sigma_c)
                # clip diagonal elements for numerical stability, per the partial implementation from Wu 2018b github repo.
                U_new = tf.matrix_set_diag(U_new, tf.clip_by_value(tf.matrix_diag_part(U_new), 1e-6, 1e8))

                return (t+1, MemoryState(R=R_new, U=U_new))

            _, self.final_state = tf.while_loop(write_loop_cond, write_loop_body, write_loop_init_vars)
            self.R = self.final_state.R 
            self.U = self.final_state.U
            self.L = tf.linalg.cholesky(self.U)

            # reading from memory: will compute elbo using q(M|X), q(w_t), memory readouts M^{T}w_t to be used by decoder.
            z_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            w_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            read_loop_init_vars = (0, z_array, w_kl_divs_array)
            read_loop_cond = lambda t, a1, a2: t < self.episode_len
            def read_loop_body(t, z_array, w_kl_divs_array):
                enc_x = self.encoded_xs[:, t, :]
                qw = self.qw(enc_x, self.R)
                w = qw.sample()

                E = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.memory_size, self.code_size])
                M = self.R if not self.sample_memory else self.R + tf.einsum('bkl,bkc->blc', self.L, E)

                z = tf.einsum('bkc,bk->bc', M, w)

                pw = self.pw(batch_size)
                kl_div_w = qw.kl_divergence(pw)

                return (t+1, z_array.write(t, z), w_kl_divs_array.write(t, kl_div_w))

            _, z_samples, w_kl_divs = tf.while_loop(read_loop_cond, read_loop_body, read_loop_init_vars)

            self.z_samples = tf.transpose(z_samples.stack(), [1, 0, 2]) # [B, T, C]
            self.w_kl_divs = tf.transpose(w_kl_divs.stack()) # [B, T]

            # decode
            self.z_samples_batched = tf.reshape(self.z_samples, [-1, self.code_size])
            self.px_z_batched = self.decode(self.z_samples_batched, training=self.is_train)                   # batch_shape [B*T], element_shape [H, W, C]
            self.px_z = tfp.distributions.BatchReshape(self.px_z_batched, batch_shape=[-1, self.episode_len]) # batch_shape [B, T], element_shape [H, W, C]
            self.log_probs_x_given_z = self.px_z.log_prob(self.input_xs)

            # dkl_M
            self.R0 = self.initial_state.R
            self.U0 = self.initial_state.U
            self.U0_inv = tf.linalg.inv(self.U0)
            term1 = tf.cast(self.code_size, dtype=tf.float32) * tf.linalg.trace(tf.einsum('bkl,blp->bkp', self.U0_inv, self.U))
            term2 = tf.linalg.trace(tf.einsum('bkc,bkz->bcz', (self.R - self.R0), tf.einsum('bkl,blc->bkc', self.U0_inv, (self.R - self.R0))))
            term3 = -tf.cast((self.memory_size * self.code_size), dtype=tf.float32)
            term4 = -2.0 * tf.cast(self.code_size, dtype=tf.float32) * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.L)), axis=-1)
            term5 = 1.0 * tf.cast(self.code_size, dtype=tf.float32) * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.U0)), axis=-1)
            self.dkl_M = 0.5 * (term1 + term2 + term3 + term4 + term5)

            # main objective
            #   note that 'log_probs_x_given_z' in the DKM context is just our naming convention and is equivalent to 'log p(x_t|w_t,M)'
            #   as the z noise variance is omitted from all readouts (Wu 2018b, Appdx A; Wu2018b, Alg. 1, etc.)
            self.elbo_episode = tf.reduce_sum(
                (self.log_probs_x_given_z - self.w_kl_divs), axis=1) - self.dkl_M  # [B]

            self.elbo = tf.reduce_mean(self.elbo_episode, axis=0)  # []
            self.elbo_per_frame = self.elbo / tf.cast(self.episode_len, dtype=tf.float32)

            self.elbo_loss = -self.elbo_per_frame

            # auxiliary objective for dkm
            self.z_ae_codes_batched = tf.reshape(self.encoded_xs, [-1, self.code_size])
            self.px_z_ae_batched = self.decode(self.z_ae_codes_batched, training=self.is_train)
            self.px_z_ae = tfp.distributions.BatchReshape(self.px_z_ae_batched, batch_shape=[-1, self.episode_len])
            self.log_probs_x_given_z_ae = self.px_z_ae.log_prob(self.input_xs)
            self.ae_reconstruction_prob_episode = tf.reduce_sum(self.log_probs_x_given_z_ae, axis=1)
            self.ae_reconstruction_prob_mean = tf.reduce_mean(self.ae_reconstruction_prob_episode, axis=0)
            self.ae_reconstruction_prob_mean_per_frame = self.ae_reconstruction_prob_mean / tf.cast(self.episode_len, dtype=tf.float32)
            self.ae_loss = -self.ae_reconstruction_prob_mean_per_frame

            # combined loss to minimize
            self.loss = self.elbo_loss + self.ae_loss

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
            self.w_kl_div_per_frame = tf.reduce_mean(self.w_kl_divs, axis=[0,1])
            self.M_kl_div_per_frame = tf.reduce_mean(self.dkl_M, axis=0) / tf.cast(self.episode_len, dtype=tf.float32)
            tf.summary.scalar('elbo_per_frame', self.elbo_per_frame)
            tf.summary.scalar('dkl_w_per_frame', self.w_kl_div_per_frame)
            tf.summary.scalar('dkm_M_per_frame', self.M_kl_div_per_frame)
            self.merged_summaries = tf.summary.merge_all()

            ## misc ops - not used during training.
            #  using a given memory state:
            self.input_R = tf.placeholder(tf.float32, shape=[None, self.memory_size, self.code_size])
            self.input_U = tf.placeholder(tf.float32, shape=[None, self.memory_size, self.memory_size])
            self.memory_state = MemoryState(R=self.input_R, U=self.input_U)

            # generate - sample p(y), p(z|y,M), p(x|z)
            def generate(M):
                batch_size = tf.shape(M)[0]
                pw = self.pw(batch_size)
                w = pw.sample()
                z = tf.einsum('bkc,bk->bc', M, w)
                px = self.decode(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x

            self.generated_x = generate(self.memory_state.R)

            # read - sample q(w|x), p(x|w,M). note that z is not a latent variable in the DKM (Appdx A).
            def read(query_x, M):
                enc_query_x = self.encode(query_x, training=False)
                qw = self.qw(enc_query_x, M)
                w = qw.sample()
                z = tf.einsum('bkc,bk->bc', M, w)
                px = self.decode(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x

            self.query_x = tf.placeholder(tf.float32, shape=[None, self.img_height, self.img_width, self.img_channels])
            self.read_x = read(self.query_x, self.memory_state.R)

            def copy(query_x):
                enc_query_x = self.encode(query_x, training=False)
                z = enc_query_x
                px = self.decode(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x

            self.copied_x = copy(self.query_x)
 
    def get_initial_state(self, batch_size):
        with tf.variable_scope('initial_state', reuse=tf.AUTO_REUSE):

            R_0_params = tf.get_variable(
                name='R_0_params', dtype=tf.float32, shape=[self.memory_size, self.code_size], 
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05), trainable=self.trainable_memory)
            # note we do not use a zero init for the DKM. 
            # this is to allow DKM RLS algo to compute a nonzero-mean addressing weight distribution
            # when using p(M) to compute the first q(w) during writing of an episode.
            # note our models use a randomized or otherwise asymmetric init for q^(0)(M) instead of assigning it the prior's values.
            # so we can (and do) use a zero init for our models' priors.

            U_0_params = tf.get_variable(
                name='U_0_params', dtype=tf.float32, shape=[self.memory_size, self.memory_size], 
                initializer=tf.zeros_initializer(), trainable=self.trainable_memory)

            R_0 = R_0_params

            upper_tri = tf.matrix_band_part(U_0_params, 0, -1)
            strictly_upper_tri = tf.matrix_set_diag(
                upper_tri, tf.zeros_like(tf.matrix_diag_part(upper_tri), dtype=tf.float32))

            logdiag = tf.matrix_diag_part(U_0_params)
            U_0_diag = tf.diag(tf.exp(logdiag))
            U_0_offdiag = strictly_upper_tri + tf.transpose(strictly_upper_tri)
            U_0 = U_0_diag + U_0_offdiag

            R = tf.tile(tf.expand_dims(R_0, 0), [batch_size, 1, 1])
            U = tf.tile(tf.expand_dims(U_0, 0), [batch_size, 1, 1])
            return MemoryState(R=R, U=U)

    def encode(self, input_x, training=True):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):

            input_x = input_x if self.discrete_outputs else (2.0 * input_x - 1.0)
            blocks = [input_x]

            for i in range(0, self.enc_blocks):
                block = self.encoder_block(blocks[i], training=training, name='enc_block_'+str(i+1))
                blocks.append(block)

            tower_output = blocks[self.enc_blocks]
            tower_output_flattened = tf.layers.flatten(tower_output)

            encoded_x = tf.layers.dense(
                tower_output_flattened, units=self.encoder_output_dim, use_bias=False, activation=None)

            return encoded_x

    def decode(self, z, training=True):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            s = (2 ** self.dec_blocks)
            dec_init_h = int(np.ceil(self.img_height / float(s)))
            dec_init_w = int(np.ceil(self.img_width / float(s)))
            fc_units = int(dec_init_h * dec_init_w * self.num_filters)

            fc = tf.layers.dense(z, units=fc_units, use_bias=False, activation=None)
            fc_3d = tf.reshape(fc, shape=[-1, dec_init_h, dec_init_w, self.num_filters])

            blocks = [fc_3d]

            for i in range(0, self.dec_blocks):
                block = self.decoder_block(blocks[i], training=training, name='dec_block_'+str(i+1))
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
            he_init = tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False)

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
            he_init = tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False)

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

    def get_qw_logsigma(self):
        with tf.variable_scope('qw_logsigma', reuse=tf.AUTO_REUSE):
            init = tf.constant_initializer(0.5 * np.log(0.3))
            qw_logsigma = tf.get_variable(
                name='qw_logsigma', dtype=tf.float32, shape=[], initializer=init, trainable=True)
            return qw_logsigma

    def qw(self, enc_x, R):
        batch_size = tf.shape(enc_x)[0]
        sigma2_z = tf.exp(2.0 * self.get_pz_logsigma())
        RRT_plus_sigma2IK = tf.einsum('bkc,blc->bkl', R, R) + sigma2_z * tf.eye(self.memory_size, batch_shape=[batch_size])
        Rz = tf.einsum('bkc,bc->bk', R, enc_x)
        mu = tf.einsum('bkl,bl->bk', tf.linalg.inv(RRT_plus_sigma2IK), Rz)
        logsigma = self.get_qw_logsigma()
        logsigma = logsigma * tf.ones_like(mu)
        w_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
        w_dist = tfp.distributions.Independent(w_dist)
        return w_dist

    def pw(self, batch_size):
        with tf.variable_scope('pw', reuse=tf.AUTO_REUSE):
            mu = tf.zeros(dtype=tf.float32, shape=[batch_size, self.memory_size])
            logsigma = tf.zeros(dtype=tf.float32, shape=[batch_size, self.memory_size])
            w_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            w_dist = tfp.distributions.Independent(w_dist)
            return w_dist

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

    def get_memory_state(self, sess, input_xs):
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: False
        }
        R, U = sess.run([self.final_state.R, self.final_state.U], feed_dict=feed_dict)
        return MemoryState(R=R, U=U)

    def generate_from_memory_state(self, sess, memory_state):
        feed_dict = {
            self.input_R: memory_state.R,
            self.input_U: memory_state.U,
            self.is_train: False
        }
        generated_x = sess.run(self.generated_x, feed_dict=feed_dict)
        return generated_x

    def read_from_memory_state(self, sess, memory_state, x):
        feed_dict = {
            self.input_R: memory_state.R,
            self.input_U: memory_state.U,
            self.query_x: x,
            self.is_train: False
        }
        read_x = sess.run(self.read_x, feed_dict=feed_dict)
        return read_x

    def copy(self, sess, x):
        feed_dict = {
            self.query_x: x,
            self.is_train: False
        }
        copied_x = sess.run(self.copied_x, feed_dict=feed_dict)
        return copied_x
