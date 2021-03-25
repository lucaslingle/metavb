import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import collections
MemoryState = collections.namedtuple('MemoryState', field_names=['R', 'U'])


class VariationalBayesianMemory:
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

        self.memory_size = hps.memory_size
        self.code_size = hps.code_size
        self.opt_iters = hps.opt_iters

        self.use_bn = hps.use_bn
        self.trainable_memory = hps.trainable_memory
        self.activation = lambda vec: tf.nn.relu(vec)

        self.lr = hps.lr

        self.global_step = tf.train.get_or_create_global_step()
        self._name = 'VBMG' if name is None else name

        with tf.variable_scope(self._name):
            self.input_xs = tf.placeholder(dtype=tf.float32, shape=[None, self.episode_len, self.img_height, self.img_width, self.img_channels])
            self.is_train = tf.placeholder(dtype=tf.bool, shape=[])

            # encode
            batch_size = tf.shape(self.input_xs)[0]
            self.input_xs_batched = tf.reshape(self.input_xs, [-1, self.img_height, self.img_width, self.img_channels])
            self.qz_batched = self.qz(self.input_xs_batched, training=self.is_train)
            self.qZ = tfp.distributions.BatchReshape(self.qz_batched, batch_shape=[-1, self.episode_len])
            self.Mu_Z = self.qZ.mean()
            self.Scale_Z_diag = self.qZ.stddev()

            self.pM = self.get_memory_prior(batch_size)

            self.R0 = self.pM.R
            self.U0 = self.pM.U
            self.U0_inv = tf.linalg.inv(self.U0)
            self.U0_inv_R0 = tf.einsum('bkl,blc->bkc', self.U0_inv, self.R0)

            ## write to memory
            self.qM_init = self.get_memory_randomized(batch_size)

            self.Mu_W_init = tf.zeros(dtype=tf.float32, shape=[batch_size, self.episode_len, self.memory_size])
            self.Sigma_w_init = tf.eye(self.memory_size, batch_shape=[batch_size])
            write_loop_init_vars = (0, self.qM_init, self.Mu_W_init, self.Sigma_w_init)
            write_loop_cond = lambda i, m, mu, sigma: i < self.opt_iters
            def write_loop_body(i, qM, Mu_W, Sigma_w):
                R = qM.R
                U = qM.U
                sigma2 = tf.stop_gradient(tf.exp(2.0 * self.get_pz_logsigma()))
                Lambda_w = tf.einsum('bkc,blc->bkl', R, R) + \
                           sigma2 * tf.eye(self.memory_size, batch_shape=[batch_size]) + \
                           tf.cast(self.code_size, dtype=tf.float32) * U
                Sigma_w = tf.linalg.inv(Lambda_w)
                Eta_W = tf.einsum('bkc,btc->btk', R, self.Mu_Z)
                Mu_W = tf.einsum('bkl,btl->btk', Sigma_w, Eta_W)

                summed_Muw_Muw_T = tf.einsum('btk,btl->bkl', Mu_W, Mu_W)
                summed_Sigmaw = tf.cast(self.episode_len, dtype=tf.float32) * Sigma_w
                Lambda_M_new = tf.stop_gradient(sigma2 * self.U0_inv) + summed_Muw_Muw_T + summed_Sigmaw
                U_new = tf.linalg.inv(Lambda_M_new)

                summed_Mu_w_Muz_T = tf.einsum('btk,btc->bkc', Mu_W, self.Mu_Z)
                Eta_M_new = tf.stop_gradient(sigma2 * self.U0_inv_R0) + summed_Mu_w_Muz_T
                R_new = tf.einsum('bkl,blc->bkc', U_new, Eta_M_new)

                return (i + 1, MemoryState(R=R_new, U=U_new), Mu_W, Sigma_w)

            _, self.qM, self.Mu_W, self.Sigma_w = tf.while_loop(
                write_loop_cond, write_loop_body, write_loop_init_vars)

            ## sample M
            self.chol_U = tf.linalg.cholesky(self.qM.U)

            ## read from memory
            z_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            z_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            w_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)

            chol_Sigma_w = tf.linalg.cholesky(self.Sigma_w)
            dkl_w_terms_134 = tf.linalg.trace(self.Sigma_w) - tf.cast(self.memory_size, dtype=tf.float32) - 2.0 * tf.reduce_sum(tf.log(tf.linalg.diag_part(chol_Sigma_w)), axis=-1)

            read_loop_init_vars = (0, z_array, z_kl_divs_array, w_kl_divs_array)
            read_loop_cond = lambda t, a1, a2, a3: t < self.episode_len
            def read_loop_body(t, z_array, z_kl_divs_array, w_kl_divs_array):
                # gaussian parameters
                Mu_z = self.Mu_Z[:, t, :]
                Scale_z_diag = self.Scale_Z_diag[:, t, :]
                qz = tfp.distributions.MultivariateNormalDiag(loc=Mu_z, scale_diag=Scale_z_diag)
                qz = tfp.distributions.Independent(qz)

                Mu_w = self.Mu_W[:, t, :]

                # samples
                z = qz.sample()

                epsilon_w = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.memory_size])
                w = Mu_w + tf.einsum('bkl,bl->bk', chol_Sigma_w, epsilon_w)

                epsilon_M = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.memory_size, self.code_size])
                M = self.qM.R + tf.einsum('bkl,blc->bkc', self.chol_U, epsilon_M)

                # dkl_w
                # computing manually instead of using qw.kl_divergence(pw). 
                # this is a performance optimization, to avoid tfp recomputing logdet(Sigma_wt) for each t in {1, ..., T}.
                # note that during writing we also avoid a similar O(TK^3) op, 
                # ... by computing the updated Sigma_wt for each t simultaneously (this value matches across timesteps!)
                dkl_w_term_2 = tf.reduce_sum(tf.square(Mu_w), axis=-1)
                dkl_w = 0.5 * (dkl_w_terms_134 + dkl_w_term_2) # kl div has four terms, and a 1/2 out front.

                # dkl_z
                # single-sample estimator of E_{M~q(M), w_t~q(w_t)}[D_{KL}(q(z_t)||p(z_t|w_t,M))].
                # found it helpful to sample M separately for each t in the episode. note that this does not change the expected value
                # note also that samplign M is expensive on its own, and if not for the savings above, training would be pretty slow.
                pz = self.pz_given_wM(w, M)
                dkl_z = qz.kl_divergence(pz)

                return (t + 1, z_array.write(t, z), z_kl_divs_array.write(t, dkl_z), w_kl_divs_array.write(t, dkl_w))

            _, z_samples, z_kl_divs, w_kl_divs = tf.while_loop(read_loop_cond, read_loop_body, read_loop_init_vars)

            self.z_samples = tf.transpose(z_samples.stack(), [1, 0, 2])  # [B, T, C]
            self.z_kl_divs = tf.transpose(z_kl_divs.stack())  # [B, T]
            self.w_kl_divs = tf.transpose(w_kl_divs.stack())  # [B, T]

            # decode
            self.z_samples_batched = tf.reshape(self.z_samples, [-1, self.code_size])  # [B*T, C]
            self.px_given_z_batched = self.px_given_z(self.z_samples_batched, training=self.is_train)  # batch_shape [B*T], element_shape [H, W, C]
            self.px_given_z_ = tfp.distributions.BatchReshape(self.px_given_z_batched, batch_shape=[-1, self.episode_len])  # batch_shape [B, T], element_shape [H, W, C]

            self.log_probs_x_given_z = self.px_given_z_.log_prob(self.input_xs)

            # dkl_M
            term1 = tf.cast(self.code_size, dtype=tf.float32) * tf.linalg.trace(tf.einsum('bkl,blp->bkp', self.U0_inv, self.qM.U))
            term2 = tf.linalg.trace(tf.einsum('bkc,bkz->bcz', (self.qM.R - self.pM.R), tf.einsum('bkl,blc->bkc', self.U0_inv, (self.qM.R - self.pM.R))))
            term3 = -tf.cast((self.memory_size * self.code_size), dtype=tf.float32)
            term4 = -2.0 * tf.cast(self.code_size, dtype=tf.float32) * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.chol_U)), axis=-1)
            term5 = 1.0 * tf.cast(self.code_size, dtype=tf.float32) * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.pM.U)), axis=-1)

            self.dkl_M = 0.5 * (term1 + term2 + term3 + term4 + term5)

            # main objective
            self.elbo_episode = tf.reduce_sum(
                (self.log_probs_x_given_z - self.z_kl_divs - self.w_kl_divs), axis=1) - self.dkl_M  # [B]

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
            self.dkl_w_per_frame = tf.reduce_mean(self.w_kl_divs, axis=[0, 1])
            self.dkl_M_per_frame = tf.reduce_mean(self.dkl_M, axis=0) / tf.cast(self.episode_len, dtype=tf.float32)
            tf.summary.scalar('elbo_per_frame', self.elbo_per_frame)
            tf.summary.scalar('dkl_z_per_frame', self.dkl_z_per_frame)
            tf.summary.scalar('dkl_w_per_frame', self.dkl_w_per_frame)
            tf.summary.scalar('dkl_M_per_frame', self.dkl_M_per_frame)
            self.merged_summaries = tf.summary.merge_all()

            ## misc ops - not used during training.
            #  using a given memory state:
            self.input_R = tf.placeholder(tf.float32, shape=[None, self.memory_size, self.code_size])
            self.input_U = tf.placeholder(tf.float32, shape=[None, self.memory_size, self.memory_size])
            self.memory_state = MemoryState(R=self.input_R, U=self.input_U)

            # generate - sample p(w), p(z|w,M), p(x|z)
            def generate(qM):
                batch_size = tf.shape(qM.R)[0]
                R = qM.R
                U = qM.U

                chol_U = tf.linalg.cholesky(U)
                epsilon_M = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.memory_size, self.code_size])
                M = R #+ tf.einsum('bkl,blc->bkc', chol_U, epsilon_M)

                pw = self.pw(batch_size=batch_size)
                w = pw.sample()
                pz = self.pz_given_wM(w, M)
                z = pz.mean()
                px = self.px_given_z(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x

            self.generated_x = generate(self.memory_state)

            # read - sample q(w|x,M), p(z|w,M), p(x|z)
            # note that q(w|x,M) is not used in training, 
            # but is equivalent to optimized q(w) for a zero-variance q(M).
            def read(query_x, qM):
                batch_size = tf.shape(qM.R)[0]
                R = qM.R
                U = qM.U
                sigma2 = tf.exp(2.0 * self.get_pz_logsigma())

                qz = self.qz(query_x, training=False)
                Mu_z = qz.mean()

                Lambda_w = tf.einsum('bkc,blc->bkl', R, R) + \
                           sigma2 * tf.eye(self.memory_size, batch_shape=[batch_size]) + \
                           tf.cast(self.code_size, dtype=tf.float32) * U
                Sigma_w = tf.linalg.inv(Lambda_w)
                Eta_w = tf.einsum('bkc,bc->bk', R, Mu_z)
                Mu_w = tf.einsum('bkl,bl->bk', Sigma_w, Eta_w)

                chol_Sigma_w = tf.linalg.cholesky(Sigma_w)
                epsilon_w = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.memory_size])
                w = Mu_w + tf.einsum('bkl,bl->bk', chol_Sigma_w, epsilon_w)

                chol_U = tf.linalg.cholesky(U)
                epsilon_M = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.memory_size, self.code_size])
                M = R + tf.einsum('bkl,blc->bkc', chol_U, epsilon_M)

                pz = self.pz_given_wM(w, M)
                z = pz.mean()
                px = self.px_given_z(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x

            self.query_x = tf.placeholder(tf.float32, shape=[None, self.img_height, self.img_width, self.img_channels])
            self.read_x = read(self.query_x, self.memory_state)

            # copy - sample q(z), p(x|z)
            def copy(query_x):
                batch_size = tf.shape(query_x)[0]
                qz = self.qz(query_x, training=False)
                z = qz.sample()
                px = self.px_given_z(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x

            self.copied_x = copy(self.query_x)

    def get_memory_prior(self, batch_size):
        with tf.variable_scope('initial_state', reuse=tf.AUTO_REUSE):

            R_0_params = tf.get_variable(
                name='R_0_params', dtype=tf.float32, shape=[self.memory_size, self.code_size], 
                initializer=tf.zeros_initializer(), trainable=self.trainable_memory)

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

    def get_memory_randomized(self, batch_size):
        # randomly initialized q(M). VBEM generally uses a randomly initialized variational posterior instead of randomizing the prior.
        # this allows symmetry breaking during inference without requiring an asymmetric prior.
        R_0 = tf.random.normal(mean=0.0, stddev=0.05, shape=[batch_size, self.memory_size, self.code_size])
        U_0 = tf.eye(self.memory_size, batch_shape=[batch_size])
        return MemoryState(R=R_0, U=U_0)

    def qz(self, input_x, training=True):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            input_x = input_x if self.discrete_outputs else (2.0 * input_x - 1.0)
            blocks = [input_x]

            for i in range(0, self.enc_blocks):
                block = self.encoder_block(blocks[i], training=training, name='enc_block_' + str(i + 1))
                blocks.append(block)

            tower_output = blocks[self.enc_blocks]
            tower_output_flattened = tf.layers.flatten(tower_output)

            encoded_x = tf.layers.dense(
                tower_output_flattened, units=(2 * self.code_size), use_bias=False, activation=None)

            fc1 = tf.layers.dense(encoded_x, units=(2 * self.code_size), use_bias=True, activation=self.activation)
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

    def get_pz_logsigma(self):
        with tf.variable_scope('pz_logsigma', reuse=tf.AUTO_REUSE):
            init = tf.zeros_initializer()
            pz_logsigma = tf.get_variable(
                name='pz_logsigma', dtype=tf.float32, shape=[], initializer=init, trainable=False)
            return pz_logsigma

    def pz_given_wM(self, w, M):
        with tf.variable_scope('pz_given_wM', reuse=tf.AUTO_REUSE):
            mu = tf.einsum('bkc,bk->bc', M, w)
            logsigma = self.get_pz_logsigma() * tf.ones_like(mu)
            z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            z_dist = tfp.distributions.Independent(z_dist)
            return z_dist

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

    def get_prior_memory_state(self, sess, input_xs):
        # uses input_xs only to get a batch size in get_initial_state method for p(M).
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: False
        }
        R, U = sess.run([self.pM.R, self.pM.U], feed_dict=feed_dict)
        return MemoryState(R=R, U=U)

    def get_posterior_memory_state(self, sess, input_xs):
        # this uses input_xs to compute variational posterior q(M)
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: False
        }
        R, U = sess.run([self.qM.R, self.qM.U], feed_dict=feed_dict)
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
