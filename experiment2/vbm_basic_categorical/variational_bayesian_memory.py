import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import collections

MemoryState = collections.namedtuple('MemoryState', field_names=['R', 'U_diagmat'])

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
        self.use_ddi = hps.use_ddi
        self.trainable_memory = hps.trainable_memory
        self.activation = lambda vec: tf.nn.relu(vec)

        self.lr = hps.lr

        self.global_step = tf.train.get_or_create_global_step()
        self._name = 'VBMC' if name is None else name

        with tf.variable_scope(self._name):
            self.input_xs_1ce = tf.placeholder(dtype=tf.float32, shape=[None, self.episode_len, self.img_height, self.img_width, self.img_channels])
            self.input_xs = tf.concat([self.input_xs_1ce, self.input_xs_1ce], axis=0) # do not reverse second copy
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
            self.U0_diagmat = self.pM.U_diagmat
            self.U0_inv_diagmat = 1.0 / self.U0_diagmat
            self.U0_inv_R0 = tf.einsum('bk,bkc->bkc', self.U0_inv_diagmat, self.pM.R) 

            ## write to memory
            self.qM_init = self.get_variational_memory_init(tf.stop_gradient(self.Mu_Z)) #data-dependent initialization. 

            self.Eta_W_init = tf.zeros(dtype=tf.float32, shape=[batch_size, self.episode_len, self.memory_size])
            self.Mu_W_init = tf.zeros(dtype=tf.float32, shape=[batch_size, self.episode_len, self.memory_size])
            write_loop_init_vars = (0, self.qM_init, self.Eta_W_init, self.Mu_W_init)
            write_loop_cond = lambda i, m, eta, mu: i < self.opt_iters
            def write_loop_body(i, qM, Eta_W, Mu_W):
                R = qM.R
                U = qM.U_diagmat

                RRT = tf.einsum('bkc,blc->bkl', R, R)
                CU = tf.cast(self.code_size, dtype=tf.float32) * U # diagmat
                diagpart_of_RRT_CU = tf.linalg.diag_part(RRT) + CU # [b,k]

                Eta_W = -0.5 * tf.expand_dims(diagpart_of_RRT_CU, 1) + tf.einsum('bkc,btc->btk', R, self.Mu_Z) # [b,t,k]
                Theta_W = tf.nn.softmax(Eta_W, axis=2)
                Mu_W = Theta_W

                Eqw_wwT_diagmat = tf.reduce_sum(Mu_W, axis=1) # shape [b,k]
                # ^ expectation happens to be a diagonal matrix. we express it as such. 

                Lambda_M_new = tf.stop_gradient(self.U0_inv_diagmat) + Eqw_wwT_diagmat
                U_new_diagmat = 1.0 / Lambda_M_new # shape [b,k]
                # ^ inverse of Lambda_M, which happens to be a diagmat

                summed_Mu_w_Muz_T = tf.einsum('btk,btc->bkc', Mu_W, self.Mu_Z)
                Eta_M_new = tf.stop_gradient(self.U0_inv_R0) + summed_Mu_w_Muz_T
                R_new = tf.einsum('bk,bkc->bkc', U_new_diagmat, Eta_M_new)

                return (i + 1, MemoryState(R=R_new, U_diagmat=U_new_diagmat), Eta_W, Mu_W)

            _, self.qM, self.Eta_W, self.Mu_W = tf.while_loop(
                write_loop_cond, write_loop_body, write_loop_init_vars)

            ## sample M
            self.chol_U_diagmat = tf.sqrt(self.qM.U_diagmat)

            ## read from memory
            z_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            z_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            w_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)

            read_loop_init_vars = (0, z_array, z_kl_divs_array, w_kl_divs_array)
            read_loop_cond = lambda t, a1, a2, a3: t < self.episode_len
            def read_loop_body(t, z_array, z_kl_divs_array, w_kl_divs_array):
                Mu_z = self.Mu_Z[:, t, :]
                Scale_z_diag = self.Scale_Z_diag[:, t, :]
                qz = tfp.distributions.MultivariateNormalDiag(loc=Mu_z, scale_diag=Scale_z_diag)
                qz = tfp.distributions.Independent(qz)

                Eta_w = self.Eta_W[:, t, :]
                Mu_w = self.Mu_W[:, t, :]

                z = qz.sample()

                epsilon_M = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.memory_size, self.code_size])
                M = self.qM.R + tf.einsum('bk,bkc->bkc', self.chol_U_diagmat, epsilon_M)

                z_kl_div_terms_array = tf.TensorArray(dtype=tf.float32, size=self.memory_size, infer_shape=True)
                z_kl_div_loop_init_vars = (0, z_kl_div_terms_array)
                z_kl_div_loop_cond = lambda k,a: k < self.memory_size
                def z_kl_div_loop_body(k, a):
                    pz = self.pz_given_wM(MTw=M[:,k,:])
                    dkl_z_contrib = Mu_w[:, k] * qz.kl_divergence(pz)
                    return (k+1, a.write(k, dkl_z_contrib))

                _, z_kl_div_terms = tf.while_loop(
                    z_kl_div_loop_cond, z_kl_div_loop_body, z_kl_div_loop_init_vars)

                qw = tfp.distributions.Independent(tfp.distributions.Categorical(logits=Eta_w))
                pw = tfp.distributions.Independent(tfp.distributions.Categorical(probs=((1.0 / float(self.memory_size)) * tf.ones_like(Eta_w))))

                dkl_w = qw.kl_divergence(pw)

                dkl_z_term_contribs = tf.transpose(z_kl_div_terms.stack(), [1, 0])  # [B, K]
                dkl_z = tf.reduce_sum(dkl_z_term_contribs, axis=-1)

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
            term1 = tf.cast(self.code_size, dtype=tf.float32) * tf.reduce_sum(tf.einsum('bk,bk->bk', self.U0_inv_diagmat, self.qM.U_diagmat), axis=-1) # C * tr[U0^-1 Uf]
            term2 = tf.linalg.trace(tf.einsum('bkc,bkz->bcz', (self.qM.R - self.pM.R), tf.einsum('bk,bkc->bkc', self.U0_inv_diagmat, (self.qM.R - self.pM.R)))) # tr[(Rf-R0)^T U0^-1 (Rf-R0)]
            term3 = -tf.cast((self.memory_size * self.code_size), dtype=tf.float32) # -KC
            term4 = -1.0 * tf.cast(self.code_size, dtype=tf.float32) * tf.reduce_sum(tf.log(self.qM.U_diagmat), axis=-1) # -C * logdet(Uf)
            term5 = 1.0 * tf.cast(self.code_size, dtype=tf.float32) * tf.reduce_sum(tf.log(self.pM.U_diagmat), axis=-1) # C * logdet(U0)

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
            self.input_U_diagmat = tf.placeholder(tf.float32, shape=[None, self.memory_size])
            self.memory_state = MemoryState(R=self.input_R, U_diagmat=self.input_U_diagmat)

            # generate - sample p(w), p(z|w,M), p(x|z)
            def generate(qM):
                batch_size = tf.shape(qM.R)[0]
                R = qM.R
                U_diagmat = qM.U_diagmat

                chol_U_diagmat = tf.sqrt(U_diagmat)
                epsilon_M = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.memory_size, self.code_size])
                M = R #+ tf.einsum('bkl,blc->bkc', chol_U_diagmat, epsilon_M)

                theta_0 = (1.0 / float(self.memory_size)) * tf.ones(dtype=tf.float32, shape=[batch_size, self.memory_size])
                pw = tfp.distributions.OneHotCategorical(probs=theta_0, dtype=tf.float32)
                pw = tfp.distributions.Independent(pw)
                w = pw.sample()
                MTw = tf.einsum('bkc,bk->bc', M, w)
                pz = self.pz_given_wM(MTw=MTw)
                z = pz.mean()
                px = self.px_given_z(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x

            self.generated_x = generate(self.memory_state)

            self.query_x = tf.placeholder(tf.float32, shape=[None, self.img_height, self.img_width, self.img_channels])

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

            U_0_diag_params = tf.get_variable(
                name='U_0_diag_params', dtype=tf.float32, shape=[self.memory_size], 
                initializer=tf.zeros_initializer(), trainable=self.trainable_memory)

            R_0 = R_0_params
            U_0_diagmat = tf.exp(U_0_diag_params)

            R = tf.tile(tf.expand_dims(R_0, 0), [batch_size, 1, 1])
            U_diagmat = tf.tile(tf.expand_dims(U_0_diagmat, 0), [batch_size, 1])

            return MemoryState(R=R, U_diagmat=U_diagmat)

    def get_variational_memory_init(self, episode_zs):
        if not self.use_ddi:
            return self.get_memory_prior(batch_size=tf.shape(episode_zs)[0])

        # variational parameters for q^{(0)}(M) initialized via random K-subset of \{\mu_{z_{t}}\}_{t=1}^{T}, 
        # which is computed from q(z_{t}).
        # for VBM-C, this form of data-dependent initialization worked better than the random data-blind initializations we tested, so we use it here.
        R = episode_zs[:, 0:self.memory_size, :] # [B, T, C] -> [B, K, C], requires K > T.
        batch_size = tf.shape(R)[0]
        U_diagmat = (1 / float(self.code_size)) * tf.ones(dtype=tf.float32, shape=[batch_size, self.memory_size]) # [B, K] because diagmat
        return MemoryState(R=R, U_diagmat=U_diagmat)


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

    def pz_given_wM(self, MTw):
        with tf.variable_scope('pz_given_wM', reuse=tf.AUTO_REUSE):
            mu = MTw
            logsigma = self.get_pz_logsigma() * tf.ones_like(mu)
            z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            z_dist = tfp.distributions.Independent(z_dist)
            return z_dist

    def train(self, sess, input_xs):
        feed_dict = {
            self.input_xs_1ce: input_xs,
            self.is_train: True
        }
        _ = sess.run(self.train_op, feed_dict=feed_dict)

        #feed_dict[self.is_train] = False
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
        R, U_diagmat = sess.run([self.pM.R, self.pM.U_diagmat], feed_dict=feed_dict)
        return MemoryState(R=R, U_diagmat=U_diagmat)

    def get_posterior_memory_state(self, sess, input_xs):
        # this uses input_xs to compute variational posterior q(M)
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: False
        }
        R, U_diagmat = sess.run([self.qM.R, self.qM.U_diagmat], feed_dict=feed_dict)
        return MemoryState(R=R, U_diagmat=U_diagmat)

    def generate_from_memory_state(self, sess, memory_state):
        feed_dict = {
            self.input_R: memory_state.R,
            self.input_U_diagmat: memory_state.U_diagmat,
            self.is_train: False
        }
        generated_x = sess.run(self.generated_x, feed_dict=feed_dict)
        return generated_x

    def copy(self, sess, x):
        feed_dict = {
            self.query_x: x,
            self.is_train: False
        }
        copied_x = sess.run(self.copied_x, feed_dict=feed_dict)
        return copied_x
