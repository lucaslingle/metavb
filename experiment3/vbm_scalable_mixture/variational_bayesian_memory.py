import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import collections

MemoryState = collections.namedtuple('MemoryState', field_names=['R', 'U', 'Mu_b', 'Sigma_b'])

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
        self.res_blocks_per_encoder_block = 1
        self.res_blocks_per_decoder_block = 1
        self.res_block_internal_dropout_rate = 0.0

        self.code_size = hps.code_size
        self.memory_size = hps.memory_size
        self.num_clusters = hps.num_clusters
        self.opt_iters = hps.opt_iters

        self.sr_alpha = hps.sr_alpha
        self.sr_beta = hps.sr_beta
        self.sr_gamma = hps.sr_gamma
        self.sr_delta = hps.sr_delta
        self.sr_epsilon = hps.sr_epsilon

        self.activation = lambda vec: tf.nn.swish(vec) # this is swish-1, see tf docs

        self.lr = hps.lr

        self.global_step = tf.train.get_or_create_global_step()
        self._name = 'VBM_GMM' if name is None else name

        with tf.variable_scope(self._name):

            # placeholders used during training and evaluation.
            self.input_xs = tf.placeholder(dtype=tf.float32, shape=[None, self.episode_len, self.img_height, self.img_width, self.img_channels])
            self.is_train = tf.placeholder(dtype=tf.bool, shape=[])
            batch_size = tf.shape(self.input_xs)[0]

            # constants.
            self.v1 = 0.40
            self.v2 = 0.40
            self.pM_sigma2 = self.memory_size * self.v1
            self.pb_sigma2 = self.memory_size * self.v2
            self.pz_sigma2 = self.memory_size * (1.0 - self.v1 - self.v2)
            self.pz_logsigma = 0.5 * tf.log(self.pz_sigma2)
            self.pb_logsigma = 0.5 * tf.log(self.pb_sigma2)

            # get memory prior p(M_1:H)p(b_1:H) = prod_h=1^H p(M_h)p(b_h).
            self.pMb = self.get_memory_prior(batch_size)

            self.R0s = self.pMb.R
            self.U0s = self.pMb.U
            self.U0_invs = tf.linalg.inv(self.U0s)
            self.U0_inv_R0s = tf.einsum('bhkl,bhlc->bhkc', self.U0_invs, self.R0s)
            self.sigma2 = tf.exp(2.0 * self.pz_logsigma)

            self.Mu_b0s = self.pMb.Mu_b
            self.Sigma_b0s = self.pMb.Sigma_b
            self.Sigma_b0_invs = tf.linalg.inv(self.Sigma_b0s)
            self.Sigma_b0_inv_Mu_b0s = tf.einsum('bhcz,bhz->bhc', self.Sigma_b0_invs, self.Mu_b0s)

            # encode
            self.input_xs_batched = tf.reshape(self.input_xs, [-1, self.img_height, self.img_width, self.img_channels])
            self.qz_batched = self.qz(self.input_xs_batched, training=self.is_train)
            self.qZ = tfp.distributions.BatchReshape(self.qz_batched, batch_shape=[-1, self.episode_len]) # [b,t], [c].
            self.Mu_Z = self.qZ.mean()
            self.Scale_diag_Z = self.qZ.stddev()

            ## write to memory using mean-field variational Bayes
            # get a good init for episode-level latents q(M_1:H)q(b_1:H).
            self.Z_for_init = self.Mu_Z
            self.qMb_init_idxs = self.compute_good_init_tf(self.Z_for_init)
            self.Mu_bs_variational_init = tf.einsum('bht,btc->bhc', self.qMb_init_idxs, self.Z_for_init)
            self.qMb_init = self.get_variational_memory_init_provided_loc(batch_size, tf.stop_gradient(self.Mu_bs_variational_init))

            # initialize local latents q(w_t)q(s_t).
            self.Mu_W_init = tf.zeros(dtype=tf.float32, shape=[batch_size, self.episode_len, self.num_clusters, self.memory_size]) # [b, t, h, k]
            self.Sigma_w_init = tf.eye(self.memory_size, batch_shape=[batch_size, self.num_clusters]) # [b, h, k, k]
            self.qs_probs_init = (1.0 / tf.cast(float(self.num_clusters), dtype=tf.float32)) * tf.ones(dtype=tf.float32, shape=[batch_size, self.episode_len, self.num_clusters]) # [b, t, h]
            write_loop_init_vars = (0, self.qMb_init, self.qs_probs_init, self.Mu_W_init, self.Sigma_w_init)
            write_loop_cond = lambda i, *_: i < self.opt_iters
            def write_loop_body(i, qMb, qs_probs, Mu_W, Sigma_w):
                
                R = qMb.R
                U = qMb.U
                Mu_b = qMb.Mu_b
                Sigma_b = qMb.Sigma_b
                tau = (1.0 / tf.stop_gradient(self.sigma2))

                qs_probs_new = self.vb_update_qs(
                    Mu_Z=self.Mu_Z, R=R, U=U, Mu_b=Mu_b, Sigma_b=Sigma_b, Mu_W=Mu_W, Sigma_w=Sigma_w, tau=tau)

                Mu_b_new, Sigma_b_new = self.vb_update_qb(
                    Mu_Z=self.Mu_Z, R=R, U=U, qs=qs_probs_new, Mu_W=Mu_W, Sigma_w=Sigma_w, tau=tau)

                Mu_W_new, Sigma_w_new = self.vb_update_qwgivens(
                    Mu_Z=self.Mu_Z, R=R, U=U, Mu_b=Mu_b_new, Sigma_b=Sigma_b_new, tau=tau)

                R_new, U_new = self.vb_update_qM(
                    Mu_Z=self.Mu_Z, Mu_b=Mu_b_new, Sigma_b=Sigma_b_new, qs=qs_probs_new, Mu_W=Mu_W_new, Sigma_w=Sigma_w_new, tau=tau)

                return (i+1, MemoryState(R=R_new, U=U_new, Mu_b=Mu_b_new, Sigma_b=Sigma_b_new), qs_probs_new, Mu_W_new, Sigma_w_new)

            _, self.qMb, self.qs, self.Mu_W, self.Sigma_w = tf.while_loop(
                write_loop_cond, write_loop_body, write_loop_init_vars)

            ## prepare cholesky decompositions to sample M, b, w. 
            self.chol_U = tf.linalg.cholesky(self.qMb.U) # [b, h, k, k]. very cheap to compute, since k is small.
            self.chol_Sigma_b = tf.linalg.cholesky(self.qMb.Sigma_b) # [b, h, c, c]. note this is always a diagmat; we should make it a vector and take the square root.
            self.chol_Sigma_w = tf.linalg.cholesky(self.Sigma_w) # [b, h, k, k]. very cheap to compute, since k is small.

            ## compute timestep-redundant terms of kl divergence for w_t
            #  will add the unique per-timestep term on later, and multiply the sum by 0.5.
            dkl_w_term_1 = tf.linalg.trace(self.Sigma_w)
            dkl_w_term_3 = -tf.cast(self.memory_size, dtype=tf.float32)
            dkl_w_term_4 = -2.0 * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.chol_Sigma_w)), axis=-1)
            dkl_w_terms_134 = dkl_w_term_1 + dkl_w_term_3 + dkl_w_term_4

            ## read from memory
            z_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            z_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            w_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            s_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)

            read_loop_init_vars = (0, z_array, z_kl_divs_array, w_kl_divs_array, s_kl_divs_array)
            read_loop_cond = lambda t, *_: t < self.episode_len
            def read_loop_body(t, z_array, z_kl_divs_array, w_kl_divs_array, s_kl_divs_array):
                # q(s_t) parameters
                qst_probs = self.qs[:, t, :]

                # q(w_t|s_t) parameters for s_t = 1, ..., H.
                Mu_wt = self.Mu_W[:, t, :, :]

                # q(z_t) parameters
                Mu_zt_perceptual = self.Mu_Z[:, t, :]
                Mu_zt_mem = tf.einsum('bh,bhc->bc', qst_probs, tf.einsum('bhkc,bhk->bhc', self.qMb.R, Mu_wt) + self.qMb.Mu_b)

                Mu_zt = self.apply_sr(Mu_zt_perceptual, Mu_zt_mem)
                Scale_diag_zt = self.Scale_diag_Z[:, t, :]

                q_zt = tfp.distributions.MultivariateNormalDiag(loc=Mu_zt, scale_diag=Scale_diag_zt)

                # sample zt for computing reconstruction likelihood.
                zt = q_zt.sample()

                # q(M), q(b), q(w_t|s_t) samples for computing z kl div.
                epsilon_M = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_clusters, self.memory_size, self.code_size]) # bhkc
                M = self.qMb.R + tf.einsum('bhkl,bhlc->bhkc', self.chol_U, epsilon_M) # bhkc
                epsilon_b = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_clusters, self.code_size]) # bhc
                b = self.qMb.Mu_b + tf.einsum('bhcz,bhz->bhc', self.chol_Sigma_b, epsilon_b) # bhc
                epsilon_wt = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_clusters, self.memory_size]) # bhk
                wt = Mu_wt + tf.einsum('bhkl,bhl->bhk', self.chol_Sigma_w, epsilon_wt) # bhk

                # compute estimator of E_{q(M)q(b)q(s_t)q(w_t|s_t)} D_{KL}(q(z_t)||p(z_t|s_t,w_t,b,M)).
                q_Mu_zt_tiled = tf.tile(tf.expand_dims(Mu_zt, 1), multiples=[1, self.num_clusters, 1])
                q_Scale_diag_zt_tiled = tf.tile(tf.expand_dims(Scale_diag_zt, 1), multiples=[1, self.num_clusters, 1])

                p_Mu_zt = tf.einsum('bhkc,bhk->bhc', M, wt) + b # bhc
                p_Scale_diag_zt = tf.exp(self.pz_logsigma) * tf.ones_like(p_Mu_zt)

                dkl_zt_term1 = tf.einsum('bhc,bhc->bh', (1.0 / tf.square(p_Scale_diag_zt)), tf.square(q_Scale_diag_zt_tiled)) # bh
                dkl_zt_term2 = tf.einsum('bhc,bhc->bh', tf.einsum('bhc,bhc->bhc', (q_Mu_zt_tiled - p_Mu_zt), (1.0 / tf.square(p_Scale_diag_zt))), (q_Mu_zt_tiled - p_Mu_zt)) # bh
                dkl_zt_term3 = -tf.cast(self.code_size, dtype=tf.float32) # scalar
                dkl_zt_term4a = -2.0 * tf.reduce_sum(tf.log(q_Scale_diag_zt_tiled), axis=-1) # bh
                dkl_zt_term4b = 2.0 * tf.reduce_sum(tf.log(p_Scale_diag_zt), axis=-1) # bh
                dkl_zt_by_cluster = 0.5 * (dkl_zt_term1 + dkl_zt_term2 + dkl_zt_term3 + dkl_zt_term4a + dkl_zt_term4b) # bh

                dkl_zt = tf.einsum('bh,bh->b', qst_probs, dkl_zt_by_cluster) # sum_{h=1}^{H} q(s_t=h) * DKL(q(z_t)||p(z_t|w_t,s_t=h,b,M)), where q(w_t|s_t=h) was already sampled from for each h.

                # compute E_{q(s_t)} D_{KL}(q(w_t|s_t)||p(w_t|s_t)).
                dkl_wt_term_2 = tf.reduce_sum(tf.square(Mu_wt), axis=-1)
                dkl_wt_by_cluster = 0.5 * (dkl_w_terms_134 + dkl_wt_term_2)
                dkl_wt = tf.einsum('bh,bh->b', qst_probs, dkl_wt_by_cluster) # sum_{h=1}^{H} q(s_t=h) * DKL(q(w_t|s_t=h)||p(w_t|s_t=h)).

                # s_kl_div computation.
                pst_probs = (1.0 / tf.cast(float(self.num_clusters), dtype=tf.float32)) * tf.ones(dtype=tf.float32, shape=[batch_size, self.num_clusters])
                dkl_st = tf.reduce_sum(qst_probs * tf.log(qst_probs), axis=1) - tf.log(pst_probs[:, 0]) # uniform prior

                return (t+1, z_array.write(t, zt), z_kl_divs_array.write(t, dkl_zt), w_kl_divs_array.write(t, dkl_wt), s_kl_divs_array.write(t, dkl_st))

            _, z_samples, z_kl_divs, w_kl_divs, s_kl_divs = tf.while_loop(read_loop_cond, read_loop_body, read_loop_init_vars)

            self.z_samples = tf.transpose(z_samples.stack(), [1, 0, 2])  # [B, T, C]
            self.z_kl_divs = tf.transpose(z_kl_divs.stack())  # [B, T]
            self.w_kl_divs = tf.transpose(w_kl_divs.stack())  # [B, T]
            self.s_kl_divs = tf.transpose(s_kl_divs.stack())  # [B, T]

            # decode
            self.z_samples_batched = tf.reshape(self.z_samples, [-1, self.code_size])  # [B*T, C]
            self.px_given_z_batched = self.px_given_z(self.z_samples_batched, training=self.is_train)  # batch_shape [B*T], element_shape [H, W, C]
            self.px_given_z_ = tfp.distributions.BatchReshape(self.px_given_z_batched, batch_shape=[-1, self.episode_len])  # batch_shape [B, T], element_shape [H, W, C]

            self.log_probs_x_given_z = self.px_given_z_.log_prob(self.input_xs)

            # dkl_M
            term1 = tf.cast(self.code_size, dtype=tf.float32) * tf.linalg.trace(tf.einsum('bhkl,bhlp->bhkp', self.U0_invs, self.qMb.U))
            term2 = tf.linalg.trace(tf.einsum('bhkc,bhkz->bhcz', (self.qMb.R - self.pMb.R), tf.einsum('bhkl,bhlc->bhkc', self.U0_invs, (self.qMb.R - self.pMb.R))))
            term3 = -tf.cast((self.memory_size * self.code_size), dtype=tf.float32)
            term4 = -2.0 * tf.cast(self.code_size, dtype=tf.float32) * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.chol_U)), axis=-1) # shape [b, h] from [b, h, k, k]
            term5 = 1.0 * tf.cast(self.code_size, dtype=tf.float32) * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.pMb.U)), axis=-1) # shape [b, h] from [b, h, k, k]
            self.dkl_M_by_cluster = 0.5 * (term1 + term2 + term3 + term4 + term5) # shape [b, h]
            self.dkl_M_total = tf.reduce_sum(self.dkl_M_by_cluster, axis=1) # shape [b]

            # dkl_b
            term1 = tf.linalg.trace(tf.einsum('bhcz,bhzq->bhcq', self.Sigma_b0_invs, self.qMb.Sigma_b))
            term2 = tf.einsum('bhc,bhc->bh', (self.qMb.Mu_b - self.pMb.Mu_b), tf.einsum('bhcz,bhz->bhc', self.Sigma_b0_invs, (self.qMb.Mu_b - self.pMb.Mu_b)))
            term3 = -tf.cast(self.code_size, dtype=tf.float32)
            term4 = -2.0 * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.chol_Sigma_b)), axis=-1) # shape [b, h] from [b, h, c, c]
            term5 = 1.0 * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.Sigma_b0s)), axis=-1) # shape [b, h] from [b, h, c, c]
            self.dkl_b_by_cluster = 0.5 * (term1 + term2 + term3 + term4 + term5) # shape [b, h]
            self.dkl_b_total = tf.reduce_sum(self.dkl_b_by_cluster, axis=1) # shape [b]

            # compute the elbo objective
            self.elbo_episode = tf.reduce_sum(
                (self.log_probs_x_given_z - self.z_kl_divs - self.w_kl_divs - self.s_kl_divs), axis=1) - self.dkl_M_total - self.dkl_b_total # [B]

            self.elbo = tf.reduce_mean(self.elbo_episode, axis=0)  # []
            self.elbo_per_frame = self.elbo / tf.cast(self.episode_len, dtype=tf.float32)

            self.loss = -self.elbo_per_frame

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.0)
            tvars = [v for v in tf.trainable_variables() if v.name.startswith(self._name)]
            self.gradients, _ = zip(*self.optimizer.compute_gradients(self.loss, tvars))
            self.gradients = [tf.where(tf.logical_or(tf.math.is_inf(g), tf.math.is_nan(g)), tf.zeros_like(v), g) for g,v in zip(self.gradients, tvars)]
            self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, 100.0)

            # use control dependencies on update ops - this is required by batchnorm.
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = self.optimizer.apply_gradients(
                    grads_and_vars=zip(self.clipped_gradients, tvars),
                    global_step=self.global_step)

            ## tensorboard summaries
            self.dkl_z_per_frame = tf.reduce_mean(self.z_kl_divs, axis=[0, 1])
            self.dkl_w_per_frame = tf.reduce_mean(self.w_kl_divs, axis=[0, 1])
            self.dkl_s_per_frame = tf.reduce_mean(self.s_kl_divs, axis=[0, 1])
            self.dkl_M_total_per_frame = tf.reduce_mean(self.dkl_M_total, axis=0) / tf.cast(self.episode_len, dtype=tf.float32)
            self.dkl_b_total_per_frame = tf.reduce_mean(self.dkl_b_total, axis=0) / tf.cast(self.episode_len, dtype=tf.float32)

            tf.summary.scalar('elbo_per_frame', self.elbo_per_frame)
            tf.summary.scalar('dkl_z_per_frame', self.dkl_z_per_frame)
            tf.summary.scalar('dkl_w_per_frame', self.dkl_w_per_frame)
            tf.summary.scalar('dkl_s_per_frame', self.dkl_s_per_frame)
            tf.summary.scalar('dkl_M_total_per_frame', self.dkl_M_total_per_frame) # this will help measure intra-cluster diversity because M controls the covariance of a cluster under the gen model.
            tf.summary.scalar('dkl_b_total_per_frame', self.dkl_b_total_per_frame)
            tf.summary.scalar('pz_sigma_squared', tf.exp(2.0 * self.pz_logsigma))
            tf.summary.scalar('pb_sigma_squared', tf.exp(2.0 * self.pb_logsigma))

            for h in range(0, self.num_clusters):
                tf.summary.scalar('cluster_{}_usage'.format(h), tf.reduce_mean(self.qs[:,:,h], axis=[0,1]))
                # this is the average of the cluster assignment value q(s_t=h) for cluster h, averaged per batch/timestep item, 
                # note this doesnt say if the clusters are actually diverse; 
                # we will use empirical variance of cluster location's variational means to measure how spread out the clusters are.

            self.cluster_posterior_loc_means_inter_cluster_mean = tf.reduce_mean(self.qMb.Mu_b, axis=1, keep_dims=True) 
            # shape [b,1,c]. this is the mean of the q(b_h)'s Mu_bhs, averaged over h, groupby episode.
            self.cluster_posterior_loc_means_empirical_variance = tf.reduce_mean(tf.square(self.qMb.Mu_b - self.cluster_posterior_loc_means_inter_cluster_mean)) 
            #shape: []. empirical variance of posterior loc means, computed per episode, groupby coord, then avgd over coords (so we have one scalar measuring inter-cluster diversity per episode)
            #which is itself averaged over episodes in a batch.

            self.cluster_prior_loc_means_inter_cluster_mean = tf.reduce_mean(self.pMb.Mu_b[0], axis=0, keep_dims=True) # shape [1,c]. this is the mean of the p(b_h)'s Mu_bhs
            self.cluster_prior_loc_means_empirical_variance = tf.reduce_mean(tf.square(self.pMb.Mu_b[0] - self.cluster_prior_loc_means_inter_cluster_mean)) 
            # shape []. empirical variance groupby coord, then avgd over coords     

            tf.summary.scalar('cluster_posterior_loc_means_empirical_variance', self.cluster_posterior_loc_means_empirical_variance)
            tf.summary.scalar('cluster_prior_loc_means_empirical_variance', self.cluster_prior_loc_means_empirical_variance)

            self.merged_summaries = tf.summary.merge_all()

            ## misc ops - not used during training.
            #  using a given memory state:
            self.input_R = tf.placeholder(tf.float32, shape=[None, self.num_clusters, self.memory_size, self.code_size])
            self.input_U = tf.placeholder(tf.float32, shape=[None, self.num_clusters, self.memory_size, self.memory_size])
            self.input_Mu_b = tf.placeholder(tf.float32, shape=[None, self.num_clusters, self.code_size])
            self.input_Sigma_b = tf.placeholder(tf.float32, shape=[None, self.num_clusters, self.code_size, self.code_size])
            self.memory_state = MemoryState(R=self.input_R, U=self.input_U, Mu_b=self.input_Mu_b, Sigma_b=self.input_Sigma_b)

            # generate - sample p(w), p(z|w,M), p(x|z)
            def generate(qMb):
                batch_size = tf.shape(qMb.R)[0]
                R = qMb.R
                U = qMb.U
                Mu_b = qMb.Mu_b
                Sigma_b = qMb.Sigma_b

                theta_0 = (1.0 / float(self.num_clusters)) * tf.ones(dtype=tf.float32, shape=[batch_size, self.num_clusters])
                ps = tfp.distributions.OneHotCategorical(probs=theta_0, dtype=tf.float32)
                s = ps.sample()
                R = tf.einsum('bh,bhkc->bkc', s, R)
                U = tf.einsum('bh,bhkl->bkl', s, U)
                Mu_b = tf.einsum('bh,bhc->bc', s, Mu_b)
                Sigma_b = tf.einsum('bh,bhcz->bcz', s, Sigma_b)    

                chol_U = tf.linalg.cholesky(U)
                epsilon_M = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.memory_size, self.code_size])
                M = R + tf.einsum('bkl,blc->bkc', chol_U, epsilon_M)

                chol_Sigma_b = tf.linalg.cholesky(Sigma_b)
                epsilon_b = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.code_size])
                b = Mu_b + tf.einsum('bcz,bz->bc', chol_Sigma_b, epsilon_b)

                pw = self.pw(batch_size=batch_size)
                w = pw.sample()
                pz = self.pz_given_wMb(w, M, b, self.pz_logsigma)
                z = pz.sample()
                px = self.px_given_z(z, training=False)
                x = px.sample()

                return x

            # generate by cluster - sample p(w), p(z|w,M), p(x|z)
            def generate_by_cluster(qMb, num_samples_per_cluster=32):
                # this is less aesthetically pleasing than ''generate'', presumably because we wrote it so all samples come from the same M_h ~ q(M_h).
                # the row-covariance seems to actually store useful information.

                # batch size should be 1 for memory state, i was lazy elsewhere. here we get rid of axis 0. 
                R = qMb.R[0]
                U = qMb.U[0]
                Mu_b = qMb.Mu_b[0]
                Sigma_b = qMb.Sigma_b[0]

                epsilon_M = tf.random_normal(mean=0.0, stddev=1.0, shape=[self.num_clusters, self.memory_size, self.code_size])
                epsilon_b = tf.random_normal(mean=0.0, stddev=1.0, shape=[self.num_clusters, self.code_size])

                chol_U = tf.linalg.cholesky(U) # hkk
                chol_Sigma_b = tf.linalg.cholesky(Sigma_b) # hcc

                M = R + tf.einsum('hkl,hlc->hkc', chol_U, epsilon_M)
                b = Mu_b + tf.einsum('hcz,hz->hc', chol_Sigma_b, epsilon_b)

                embs = tf.eye(self.num_clusters) # shape [H, E]. one-hot embs for each cluster, indexed by cluster id. 
                embs = tf.tile(tf.expand_dims(embs, 1), multiples=[1, num_samples_per_cluster, 1]) # shape [H, N, E]
                embs = tf.reshape(embs, [-1, self.num_clusters]) # shape [H*N, E]. same emb for N consecutive idxs on axis 0.

                s = embs
                batch_size = tf.shape(s)[0]

                M = tf.einsum('bh,hkc->bkc', s, M)
                b = tf.einsum('bh,hc->bc', s, b)

                pw = self.pw(batch_size=batch_size)
                w = pw.sample()
                pz = self.pz_given_wMb(w, M, b, self.pz_logsigma)
                z = pz.sample()
                px = self.px_given_z(z, training=False)
                x = px.sample()

                return x

            self.generated_x = generate(self.memory_state)
            self.generated_x_by_cluster = generate_by_cluster(self.memory_state)

            # read - given q(z) perceptual, infer q(s), q(w|s); then sample q(s), q(w|s), q(M), q(b), p(z|w, M_s, b_s), p(x|z)
            def read(query_x, qMb):
                batch_size = tf.shape(qMb.R)[0]
                R = qMb.R
                U = qMb.U
                Mu_b = qMb.Mu_b
                Sigma_b = qMb.Sigma_b

                qz_perceptual = self.qz(query_x, training=False)
                Mu_z_perceptual = qz_perceptual.mean()

                # optimize variational distributions q(s)q(w) for local latent variables y = {s, w} for reading from memory Omega = {M_1:H, b_1:H}. 
                Mu_w_init = tf.zeros(dtype=tf.float32, shape=[batch_size, self.num_clusters, self.memory_size])
                Sigma_w_init = tf.eye(self.memory_size, batch_shape=[batch_size, self.num_clusters])
                qs_probs_init = (1.0 / float(self.num_clusters)) * tf.ones(dtype=tf.float32, shape=[batch_size, self.num_clusters])
                C = tf.cast(self.code_size, dtype=tf.float32)
                tau = (1.0 / tf.stop_gradient(self.sigma2))

                def addressing_loop_body(i, qs_probs, Mu_w, Sigma_w):
                    RRT_plus_CU = tf.einsum('bhkc,bhlc->bhkl', R, R) + C * U
                    qs_logits_term_1a = tau * tf.einsum('bhk,bhk->bh', Mu_w, tf.einsum('bhkl,bhl->bhk', RRT_plus_CU, Mu_w))
                    qs_logits_term_1b = tau * tf.linalg.trace(tf.einsum('bhkl,bhlp->bhkp', RRT_plus_CU, Sigma_w))
                    qs_logits_term_1c = tau * tf.einsum('bhc,bhc->bh', Mu_b, Mu_b)
                    qs_logits_term_1d = tau * tf.linalg.trace(Sigma_b)
                    qs_logits_term_1 = -0.5 * (qs_logits_term_1a + qs_logits_term_1b + qs_logits_term_1c + qs_logits_term_1d)
    
                    Mu_z_minus_Mu_bh = tf.expand_dims(Mu_z_perceptual, 1) - Mu_b # bhc
                    qs_logits_term_2 = tau * tf.einsum('bhk,bhk->bh', Mu_w, tf.einsum('bhkc,bhc->bhk', R, Mu_z_minus_Mu_bh))
                    qs_logits_term_3 = tau * tf.einsum('bhc,bc->bh', Mu_b, Mu_z_perceptual)
                    qs_logits_term_4 = 0.5 * (2 * tf.reduce_sum(tf.log(tf.linalg.diag_part(tf.linalg.cholesky(Sigma_w))), axis=-1))
                    qs_logits_term_5 = -0.5 * (tf.einsum('bhk,bhk->bh', Mu_w, Mu_w) + tf.linalg.trace(Sigma_w))
                    qs_logits = qs_logits_term_1 + qs_logits_term_2 + qs_logits_term_3 + qs_logits_term_4 + qs_logits_term_5
                    qs_probs_new = tf.nn.softmax(qs_logits, axis=-1)
                    qs_probs_new = (1e-3 + qs_probs_new) / tf.reduce_sum((1e-3 + qs_probs_new), axis=-1, keep_dims=True)

                    Lambda_w_new = tf.eye(self.memory_size, batch_shape=[batch_size, self.num_clusters]) + \
                           tf.exp(-2.0 * self.pz_logsigma) * tf.einsum('bhkc,bhlc->bhkl', R, R) + \
                           tf.exp(-2.0 * self.pz_logsigma) * tf.cast(self.code_size, dtype=tf.float32) * U
                    Sigma_w_new = tf.linalg.inv(Lambda_w_new)
                    Eta_w_new = tf.exp(-2.0 * self.pz_logsigma) * tf.einsum('bhkc,bhc->bhk', R, (Mu_z_perceptual - Mu_b))
                    Mu_w_new = tf.einsum('bhkl,bhl->bhk', Sigma_w_new, Eta_w_new)

                    return (i+1, qs_probs_new, Mu_w_new, Sigma_w_new)

                addressing_loop_init = (0, qs_probs_init, Mu_w_init, Sigma_w_init)
                addressing_loop_cond = lambda i, *_: i < self.opt_iters
                _, qs_probs, Mu_w, Sigma_w = tf.while_loop(addressing_loop_cond, addressing_loop_body, addressing_loop_init)

                qs = tfp.distributions.OneHotCategorical(probs=qs_probs, dtype=tf.float32)
                s = qs.sample()

                chol_Sigma_w = tf.linalg.cholesky(Sigma_w)
                epsilon_w = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_clusters, self.memory_size])
                w = Mu_w + tf.einsum('bhkl,bhl->bhk', chol_Sigma_w, epsilon_w)

                chol_U = tf.linalg.cholesky(U)
                epsilon_M = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_clusters, self.memory_size, self.code_size])
                M = R + tf.einsum('bhkl,bhlc->bhkc', chol_U, epsilon_M)

                chol_Sigma_b = tf.linalg.cholesky(Sigma_b)
                epsilon_b = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_clusters, self.code_size])
                b = Mu_b + tf.einsum('bhcz,bhz->bhc', chol_Sigma_b, epsilon_b)

                w = tf.einsum('bh,bhk->bk', s, w)
                M = tf.einsum('bh,bhkc->bkc', s, M)
                b = tf.einsum('bh,bhc->bc', s, b)

                pz = self.pz_given_wMb(w, M, b, self.pz_logsigma)
                z = pz.sample()

                px = self.px_given_z(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x, z

            self.query_x = tf.placeholder(tf.float32, shape=[None, self.img_height, self.img_width, self.img_channels])
            self.read_x, self.read_z = read(self.query_x, self.memory_state)

            # copy - sample q(z), p(x|z)
            def copy(query_x):
                batch_size = tf.shape(query_x)[0]
                qz = self.qz(query_x, training=False)
                z = qz.mean() + tf.sqrt(self.pz_sigma2) * tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.code_size])
                px = self.px_given_z(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x, z

            self.copied_x, self.copied_z = copy(self.query_x)

    def get_memory_prior(self, batch_size):
        with tf.variable_scope('initial_state', reuse=tf.AUTO_REUSE):

            R_0 = tf.get_variable(
                name='R_0', dtype=tf.float32, shape=[self.memory_size, self.code_size], 
                initializer=tf.zeros_initializer(), trainable=False) # shape hkc
            U_0 = (self.pM_sigma2 / tf.cast(float(self.memory_size), dtype=tf.float32)) * tf.eye(self.memory_size) # shape hkk

            Mu_b0 = tf.get_variable(name='Mu_b0', dtype=tf.float32, shape=[self.code_size],
                initializer=tf.zeros_initializer(), trainable=False) # shape hc
            Sigma_b0 = self.pb_sigma2 * tf.eye(self.code_size) # shape hcc

            # this code is to support dynamically resizable memory.
            # i.e., it allows a variable number of clusters during deployment w/o tensor restore issues.
            R_0 = tf.reshape(R_0, [1, 1, self.memory_size, self.code_size])
            U_0 = tf.reshape(U_0, [1, 1, self.memory_size, self.memory_size])
            Mu_b0 = tf.reshape(Mu_b0, [1, 1, self.code_size])
            Sigma_b0 = tf.reshape(Sigma_b0, [1, 1, self.code_size, self.code_size]) # todo: all Sigma_b's can and should be replaced w/ a diagmat

            R_0 = tf.tile(R_0, [batch_size, self.num_clusters, 1, 1]) # shape bhkc
            U_0 = tf.tile(U_0, [batch_size, self.num_clusters, 1, 1]) # shape bhkk
            Mu_b0 = tf.tile(Mu_b0, [batch_size, self.num_clusters, 1]) # shape bhc 
            Sigma_b0 = tf.tile(Sigma_b0, [batch_size, self.num_clusters, 1, 1]) # shape bhcc

            return MemoryState(R=R_0, U=U_0, Mu_b=Mu_b0, Sigma_b=Sigma_b0)

    def get_variational_memory_init(self, batch_size):
        # randomly initialized q(M1:H)q(b1:H). 
        # VBEM generally uses a randomly initialized variational posterior instead of randomizing the prior.
        # this allows symmetry breaking during inference without requiring an asymmetric prior.
        # this initialization is a highly heuristic one.
        R = tf.random.normal(mean=0.0, stddev=(1.0 / float(self.code_size)), shape=[batch_size, self.num_clusters, self.memory_size, self.code_size]) # shape bhkc
        U =  (1.0 / float(self.code_size)) * tf.eye(self.memory_size, batch_shape=[batch_size, self.num_clusters]) # shape bhkk
        Mu_b = tf.random.normal(mean=0.0, stddev=tf.stop_gradient(tf.exp(self.pb_logsigma)), shape=[batch_size, self.num_clusters, self.code_size]) # shape bhc
        Sigma_b = tf.stop_gradient(tf.exp(2.0 * self.pb_logsigma)) * tf.eye(self.code_size, batch_shape=[batch_size, self.num_clusters]) # shape bhcc
        return MemoryState(R=R, U=U, Mu_b=Mu_b, Sigma_b=Sigma_b)

    def get_variational_memory_init_provided_loc(self, batch_size, loc_means):
        # randomly initialized q(M1:H)q(b1:H). VBEM generally uses a randomly initialized variational posterior instead of randomizing the prior.
        # this allows symmetry breaking during inference without requiring an asymmetric prior.
        # this initialization is a highly heuristic one.
        R = tf.random.normal(mean=0.0, stddev=(1.0 / float(self.code_size)), shape=[batch_size, self.num_clusters, self.memory_size, self.code_size]) # shape bhkc
        U = (1.0 / float(self.code_size)) * tf.eye(self.memory_size, batch_shape=[batch_size, self.num_clusters]) # shape bhkk
        Mu_b = tf.stop_gradient(loc_means) # shape bhc
        Sigma_b = tf.stop_gradient(tf.exp(2.0 * self.pb_logsigma)) * tf.eye(self.code_size, batch_shape=[batch_size, self.num_clusters]) # shape bhcc
        return MemoryState(R=R, U=U, Mu_b=Mu_b, Sigma_b=Sigma_b)

    def get_variational_memory_init_biglocinit(self, batch_size):
        R = tf.random.normal(mean=0.0, stddev=(1.0 / float(self.code_size)), shape=[batch_size, self.num_clusters, self.memory_size, self.code_size]) # shape bhkc
        U =  (1.0 / float(self.code_size)) * tf.eye(self.memory_size, batch_shape=[batch_size, self.num_clusters]) # shape bhkk
        Mu_b = tf.random.normal(mean=0.0, stddev=tf.stop_gradient(tf.sqrt(self.pb_sigma2 + self.pM_sigma2 + self.pz_sigma2)), shape=[batch_size, self.num_clusters, self.code_size]) # shape bhc
        Sigma_b = tf.stop_gradient(tf.exp(2.0 * self.pb_logsigma)) * tf.eye(self.code_size, batch_shape=[batch_size, self.num_clusters]) # shape bhcc
        return MemoryState(R=R, U=U, Mu_b=Mu_b, Sigma_b=Sigma_b)

    def vb_update_qs(self, Mu_Z, R, U, Mu_b, Sigma_b, Mu_W, Sigma_w, tau):
        # computes vb update for q(s_t).
        C = tf.cast(self.code_size, dtype=tf.float32)
        RRT_plus_CU = tf.einsum('bhkc,bhlc->bhkl', R, R) + C * U # RhRh^T + CUh

        qs_logits_term_1a = tau * tf.einsum('bthk,bthk->bth', Mu_W, tf.einsum('bhkl,bthl->bthk', RRT_plus_CU, Mu_W)) # shape: [bth]
        qs_logits_term_1b = tau * tf.linalg.trace(tf.einsum('bhkl,bthlp->bthkp', RRT_plus_CU, tf.expand_dims(Sigma_w, 1))) # shape: [b1h]
        qs_logits_term_1c = tau * tf.expand_dims(tf.einsum('bhc,bhc->bh', Mu_b, Mu_b), 1) # shape: [b1h]. 
        qs_logits_term_1d = tau * tf.expand_dims(tf.linalg.trace(Sigma_b), 1) # shape: [b1h]. 
        qs_logits_term_1 = -0.5 * (qs_logits_term_1a + qs_logits_term_1b + qs_logits_term_1c + qs_logits_term_1d) # shape [bth]

        Mu_Z_minus_Mu_bh = tf.expand_dims(Mu_Z, 2) - tf.expand_dims(Mu_b, 1) # shape: [bthc]. 
        qs_logits_term_2 = tau * tf.einsum('bthk,bthk->bth', Mu_W, tf.einsum('bhkc,bthc->bthk', R, Mu_Z_minus_Mu_bh)) # shape: [bth]. 
        qs_logits_term_3 = tau * tf.einsum('bhc,btc->bth', Mu_b, Mu_Z) # shape: [bth]. 
        qs_logits_term_4 = tf.expand_dims(2 * tf.reduce_sum(tf.log(1e-6 + tf.linalg.diag_part(tf.linalg.cholesky(Sigma_w))), axis=-1), 1) # shape: [b1h]. 
        qs_logits_term_5 = -0.5 * (tf.einsum('bthk,bthk->bth', Mu_W, Mu_W) + tf.expand_dims(tf.linalg.trace(Sigma_w), 1)) # shape: [bth]. 
        
        qs_logits = qs_logits_term_1 + qs_logits_term_2 + qs_logits_term_3 + qs_logits_term_4 + qs_logits_term_5
        qs_probs_new = tf.nn.softmax(qs_logits, axis=-1) # shape [bth]
        qs_probs_new = (1e-3 + qs_probs_new) / tf.reduce_sum((1e-3 + qs_probs_new), axis=-1, keep_dims=True)

        return qs_probs_new

    def vb_update_qb(self, Mu_Z, R, U, qs, Mu_W, Sigma_w, tau):
        # computes vb update for q(b_h).
        batch_size = tf.shape(Mu_Z)[0]

        I_C = tf.eye(self.code_size, batch_shape=[batch_size, self.num_clusters])
        Lambda_b_scalars = (1.0 / self.pb_sigma2) + tau * tf.reduce_sum(qs, axis=1) # shape: [bh].
        Sigma_b_scalars = (1.0 / Lambda_b_scalars)
        Sigma_b_new = tf.einsum('bh,bhcz->bhcz', Sigma_b_scalars, I_C)

        Muzt_minus_RT_Muwt = tf.expand_dims(self.Mu_Z, 2) - tf.einsum('bhkc,bthk->bthc', R, Mu_W) # shape: [bthc].
        weighted_summed_Muzt_minus_RT_Muwt = tf.einsum('bth,bthc->bhc', qs, Muzt_minus_RT_Muwt) # shape: [bhc].
        Eta_b_new = tf.stop_gradient(self.Sigma_b0_inv_Mu_b0s) + tau * weighted_summed_Muzt_minus_RT_Muwt # shape: [bhc].
        Mu_b_new = tf.einsum('bhcz,bhz->bhc', Sigma_b_new, Eta_b_new)

        return Mu_b_new, Sigma_b_new

    def vb_update_qwgivens(self, Mu_Z, R, U, Mu_b, Sigma_b, tau):
        # computes vb update for q(w_t|s_t).
        batch_size = tf.shape(Mu_Z)[0]
        C = tf.cast(self.code_size, dtype=tf.float32)

        I_K = tf.eye(self.memory_size, batch_shape=[batch_size, self.num_clusters])
        RRT_plus_CU = tf.einsum('bhkc,bhlc->bhkl', R, R) + C * U # shape [bhkl]. 
        Lambda_w_new = I_K + tau * RRT_plus_CU
        Sigma_w_new = tf.linalg.inv(Lambda_w_new)

        Muzt_minus_Mubh = tf.expand_dims(Mu_Z, 2) - tf.expand_dims(Mu_b, 1) # shape: [bthc].
        Eta_W_new = tau * tf.einsum('bhkc,bthc->bthk', R, Muzt_minus_Mubh)
        Mu_W_new = tf.einsum('bhkl,bthl->bthk', Sigma_w_new, Eta_W_new)

        return Mu_W_new, Sigma_w_new

    def vb_update_qM(self, Mu_Z, Mu_b, Sigma_b, qs, Mu_W, Sigma_w, tau):
        # computes vb update for q(M_h).
        weighted_summed_MuwMuwT = tf.einsum('bthk,bthl->bhkl', tf.einsum('bth,bthk->bthk', qs, Mu_W), Mu_W) # more efficient 
        weighted_summed_Sigmaw = tf.einsum('bh,bhkl->bhkl', tf.reduce_sum(qs, axis=1), Sigma_w) # more efficient
        weighted_summed_MuwMuwTplusSigmaw = weighted_summed_MuwMuwT + weighted_summed_Sigmaw
        Lambda_M_new = tf.stop_gradient(self.U0_invs) + tau * weighted_summed_MuwMuwTplusSigmaw
        U_new = tf.linalg.inv(Lambda_M_new)

        weighted_Muw = tf.einsum('bth,bthk->bthk', qs, Mu_W)
        weighted_summed_MuwMuzT = tf.einsum('bthk,btc->bhkc', weighted_Muw, self.Mu_Z)
        weighted_summed_MuwMubhT = tf.einsum('bthk,bhc->bhkc', weighted_Muw, Mu_b)
        weighted_summed_Muw_outer_MuzminusMubT = weighted_summed_MuwMuzT - weighted_summed_MuwMubhT

        Eta_M_new = tf.stop_gradient(self.U0_inv_R0s) + tau * weighted_summed_Muw_outer_MuzminusMubT
        R_new = tf.einsum('bhkl,bhlc->bhkc', U_new, Eta_M_new)
        return R_new, U_new

    def qz(self, input_x, training=True):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            input_x = input_x if self.discrete_outputs else (2.0 * input_x - 1.0)
            blocks = [input_x]

            for i in range(0, self.enc_blocks):
                block = self.encoder_block(blocks[i], training=training, name='enc_block_' + str(i + 1))
                blocks.append(block)

            last = blocks[self.enc_blocks]
            last = tf.layers.flatten(last)
            encoded_x = tf.layers.dense(
                last, units=(2 * self.code_size), use_bias=False, activation=None) 

            fc1 = tf.layers.dense(encoded_x, units=(2 * self.code_size), use_bias=True, activation=self.activation)
            fc2 = tf.layers.dense(fc1, units=self.code_size, use_bias=False, activation=None)
            mu = fc2
            scale_diag = tf.sqrt(0.5 * self.pz_sigma2) * tf.ones_like(mu)
            z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=scale_diag)
            z_dist = tfp.distributions.Independent(z_dist)
            return z_dist

    def px_given_z(self, z, training=True):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            s = (2 ** self.dec_blocks)
            dec_init_h = int(np.ceil(self.img_height / float(s)))
            dec_init_w = int(np.ceil(self.img_width / float(s)))
            fc_units = int(dec_init_h * dec_init_w * self.num_filters)

            fc = tf.layers.dense(z, units=fc_units, use_bias=False, activation=None, kernel_initializer=tf.orthogonal_initializer())
            fc_3d = tf.reshape(fc, shape=[-1, dec_init_h, dec_init_w, self.num_filters])

            blocks = [fc_3d]
            for i in range(0, self.dec_blocks):
                block = self.decoder_block(blocks[i], training=training, name='dec_block_' + str(i + 1))
                blocks.append(block)

            last = blocks[self.dec_blocks]
            last = self.maybe_center_crop(last, self.img_height, self.img_width)

            if self.discrete_outputs:
                # this is to support binarized image data only
                decoded_logits_x = tf.layers.conv2d(
                    last, filters=self.img_channels,
                    kernel_size=1, strides=1, padding='same', activation=None)

                x_dist = tfp.distributions.Bernoulli(logits=decoded_logits_x)
                x_dist = tfp.distributions.Independent(x_dist)
                return x_dist
            else:
                # this is stable, and worked better than using a non-residual block of stride > 1.  
                decoded_mu_x = tf.layers.conv2d(
                    last, filters=self.img_channels,
                    kernel_size=1, strides=1, padding='same', activation=tf.nn.sigmoid)

                decoded_sigma_x = 5e-3 + 0.50 * tf.layers.conv2d(
                    last, filters=self.img_channels,
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

    def encoder_block(self, inputs, training, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            glorot_init = tf.glorot_normal_initializer()

            conv = tf.layers.conv2d(
                inputs, filters=self.num_filters, kernel_size=4, strides=2, padding='valid', 
                kernel_initializer=glorot_init, activation=None)

            res_blocks = [conv]
            for j in range(0, self.res_blocks_per_encoder_block):
                res_block = self.residual_block(res_blocks[j], training=training, name='res_block_' + str(j+1))
                res_blocks.append(res_block)

            last = res_blocks[self.res_blocks_per_encoder_block]
            output = last

            return output

    def group_norm(self, inputs, num_groups, name):
        channels = inputs.get_shape().as_list()[-1]
        assert channels % num_groups == 0

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            group_size = channels // num_groups
            loc = tf.get_variable(name='loc', shape=[num_groups, group_size], initializer=tf.constant_initializer(value=0.0))
            scale = tf.get_variable(name='scale', shape=[num_groups, group_size], initializer=tf.constant_initializer(value=1.0))

            good_shape = [-1] + inputs.get_shape().as_list()[1:3] + [num_groups, (channels // num_groups)]
            inputs_by_group = tf.reshape(inputs, good_shape) # [b,h,w,g,f]
            means_by_group = tf.reduce_mean(inputs_by_group, axis=[1,2,4], keep_dims=True)
            zeromean_inputs_by_group = inputs_by_group - means_by_group
            stddev_by_group = tf.sqrt(1e-6 + tf.reduce_mean(tf.square(zeromean_inputs_by_group), axis=[1,2,4], keep_dims=True))
            standardized_inputs_by_group = zeromean_inputs_by_group / stddev_by_group

            broadcast_shape = [1, 1, 1] + [num_groups, group_size]
            loc = tf.reshape(loc, broadcast_shape) # broadcast for dynamic batch size
            scale = tf.reshape(scale, broadcast_shape)
            groupnormed_inputs_by_group = scale * standardized_inputs_by_group + loc

            target_shape = [-1] + inputs.get_shape().as_list()[1:]
            groupnormed_inputs = tf.reshape(groupnormed_inputs_by_group, target_shape) # shape [b,h,w,gf] where the i-th set of f consecutive entries on channel axis will correspond to group i.
            return groupnormed_inputs

    def residual_block(self, inputs, training, name):
        ## residual block without bottleneck
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)

            conv1 = tf.layers.conv2d(
                inputs, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            norm1 = self.group_norm(conv1, num_groups=(self.num_filters // 4), name='gn1')

            act1 = self.activation(norm1)
            drop1 = tf.layers.dropout(act1, rate=self.res_block_internal_dropout_rate, training=training)

            conv2 = tf.layers.conv2d(
                drop1, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            norm2 = self.group_norm(conv2, num_groups=(self.num_filters // 4), name='gn2')

            act2 = self.activation(inputs + norm2)
            output = act2

            return output

    def decoder_block(self, inputs, training, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            glorot_init = tf.glorot_normal_initializer()

            res_blocks = [inputs]
            for j in range(0, self.res_blocks_per_decoder_block):
                res_block = self.residual_block(res_blocks[j], training=training, name='res_block_' + str(j+1))
                res_blocks.append(res_block)

            last = res_blocks[self.res_blocks_per_decoder_block]

            dconv = tf.layers.conv2d_transpose(
                last, filters=self.num_filters, kernel_size=4, strides=2, padding='same',
                kernel_initializer=glorot_init, activation=None)
            output = dconv

            return output

    def apply_sr(self, Mu_z_perceptual, Mu_z_mem, training=True):
        # apply the novel stochastic regularizer.
        assert self.sr_delta == 2 * self.sr_epsilon and self.sr_epsilon >= np.finfo(np.float32).eps
        batch_size = tf.shape(Mu_z_perceptual)[0]
        alpha_vec = self.sr_alpha * tf.ones(dtype=tf.float32, shape=[batch_size, 1])
        beta_vec = self.sr_beta * tf.ones(dtype=tf.float32, shape=[batch_size, 1])
        gamma_vec = self.sr_gamma * tf.ones(dtype=tf.float32, shape=[batch_size, 1])
        delta_vec = self.sr_delta * tf.ones(dtype=tf.float32, shape=[batch_size, 1])
        epsilon_vec = self.sr_epsilon * tf.ones(dtype=tf.float32, shape=[batch_size, 1])

        s_dist = tfp.distributions.Beta(concentration1=alpha_vec, concentration0=beta_vec) # this is Beta(alpha, beta), see tfp docs.
        s = s_dist.sample()
        left = gamma_vec - epsilon_vec
        t = left + delta_vec * s
        Mu_z = t * Mu_z_perceptual + (1. - t) * Mu_z_mem
        return Mu_z

    def pz_given_wMb(self, w, M, b, pz_logsigma):
        with tf.variable_scope('pz_given_wM', reuse=tf.AUTO_REUSE):
            mu = tf.einsum('bkc,bk->bc', M, w) + b
            logsigma = pz_logsigma * tf.ones_like(mu)
            z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            return z_dist

    def pw(self, batch_size):
        with tf.variable_scope('pw', reuse=tf.AUTO_REUSE):
            mu = tf.zeros(dtype=tf.float32, shape=[batch_size, self.memory_size])
            logsigma = tf.zeros(dtype=tf.float32, shape=[batch_size, self.memory_size])
            w_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            return w_dist

    def compute_good_init_tf(self, episode_zs):
        # this implements the k-means++ initialization algorithm (Arthur and Vassilvitskii, 2007)
        # within tensorflow rather than require two session.run calls. 
        # we do not run Lloyd's iterations, only the seeding algorithm.
        batch_size = tf.shape(episode_zs)[0]
        good_init_idxs_onehot_array = tf.TensorArray(dtype=tf.float32, size=self.num_clusters, infer_shape=True)
        uniform_probs = (1.0/tf.cast(float(self.episode_len), dtype=tf.float32)) * tf.ones(dtype=tf.float32, shape=[batch_size, self.episode_len])
        uniform_onehot = tfp.distributions.OneHotCategorical(probs=uniform_probs, dtype=tf.float32)
        first = uniform_onehot.sample()
        good_init_idxs_onehot_array = good_init_idxs_onehot_array.write(0, first)
        min_squared_distances_from_selecteds = tf.reduce_sum(tf.square(tf.expand_dims(tf.einsum('bt,btc->bc', first, episode_zs), 1) - episode_zs), axis=-1) # bt

        loop_init_vars = (1, min_squared_distances_from_selecteds, good_init_idxs_onehot_array)
        loop_cond = lambda i, *_: i < self.num_clusters
        def loop_body(i, min_squared_distances_from_selecteds, array):
            probs = min_squared_distances_from_selecteds / tf.reduce_sum(min_squared_distances_from_selecteds, axis=-1, keep_dims=True) # shape: [B, T]
            distribution = tfp.distributions.OneHotCategorical(probs=probs, dtype=tf.float32)
            selected_onehot = distribution.sample()
            selected_code = tf.einsum('bt,btc->bc', selected_onehot, episode_zs)
            squared_distances_from_selected_code = tf.reduce_sum(tf.square(tf.expand_dims(selected_code, 1) - episode_zs), axis=-1) # bt
            min_squared_distances_from_selecteds_new = tf.minimum(min_squared_distances_from_selecteds, squared_distances_from_selected_code)
            return (i+1, min_squared_distances_from_selecteds_new, array.write(i, selected_onehot))

        _, _, good_init_idxs_onehot_array_final = tf.while_loop(loop_cond, loop_body, loop_init_vars)
        good_init_idxs_onehot_final = tf.transpose(good_init_idxs_onehot_array_final.stack(), perm=[1,0,2]) # [b,h,t]
        result = tf.stop_gradient(good_init_idxs_onehot_final)
        return result

    def train(self, sess, input_xs):
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: True
        }
        fetch_list = [
            self.train_op, 
            self.elbo_per_frame, 
            self.global_step, 
            self.merged_summaries
        ]
        _, elbo_per_frame, step, summaries = sess.run(fetch_list, feed_dict=feed_dict)
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
        R, U, Mu_b, Sigma_b = sess.run([self.pMb.R, self.pMb.U, self.pMb.Mu_b, self.pMb.Sigma_b], feed_dict=feed_dict)
        return MemoryState(R=R, U=U, Mu_b=Mu_b, Sigma_b=Sigma_b)

    def get_posterior_memory_state(self, sess, input_xs):
        # this uses input_xs to compute variational posterior q(M)
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: False
        }
        R, U, Mu_b, Sigma_b = sess.run([self.qMb.R, self.qMb.U, self.qMb.Mu_b, self.qMb.Sigma_b], feed_dict=feed_dict)
        return MemoryState(R=R, U=U, Mu_b=Mu_b, Sigma_b=Sigma_b)

    def generate_from_memory_state(self, sess, memory_state):
        feed_dict = {
            self.input_R: memory_state.R,
            self.input_U: memory_state.U,
            self.input_Mu_b: memory_state.Mu_b,
            self.input_Sigma_b: memory_state.Sigma_b,
            self.is_train: False
        }
        generated_x = sess.run(self.generated_x, feed_dict=feed_dict)
        return generated_x

    def generate_by_cluster(self, sess, memory_state):
        feed_dict = {
            self.input_R: memory_state.R,
            self.input_U: memory_state.U,
            self.input_Mu_b: memory_state.Mu_b,
            self.input_Sigma_b: memory_state.Sigma_b,
            self.is_train: False
        }
        generated_x_by_cluster = sess.run(self.generated_x_by_cluster, feed_dict=feed_dict)
        return generated_x_by_cluster

    def read_from_memory_state(self, sess, memory_state, x):
        feed_dict = {
            self.input_R: memory_state.R,
            self.input_U: memory_state.U,
            self.input_Mu_b: memory_state.Mu_b,
            self.input_Sigma_b: memory_state.Sigma_b,
            self.query_x: x,
            self.is_train: False
        }
        read_x, read_z = sess.run([self.read_x, self.read_z], feed_dict=feed_dict)
        return read_x, read_z

    def copy(self, sess, x):
        feed_dict = {
            self.query_x: x,
            self.is_train: False
        }
        copied_x, copied_z = sess.run([self.copied_x, self.copied_z], feed_dict=feed_dict)
        return copied_x, copied_z
